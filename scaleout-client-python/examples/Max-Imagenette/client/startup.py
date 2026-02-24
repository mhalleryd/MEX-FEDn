import os
import random
import sys
import math
import threading
import time

import torch
import numpy as np

from .data import load_data, prepare_data
from .model import load_parameters, save_parameters

from scaleout import EdgeClient
from scaleoututil.utils.model import ScaleoutModel

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def startup(client: EdgeClient):
    """Entry point called by Scaleout Edge."""
    prepare_data()
    client = MyClient(client)
    client.start_log_attributes()


class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = None

        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
        client.set_predict_callback(self.predict)
    
    def start_log_attributes(self):
        """Start a background thread that logs attributes periodically."""
        t = threading.Thread(target=self._log_loop, daemon=True)
        t.start()

    # log randomly charing or not. The server functions implementation will only train on clients that are currently charging.
    def _log_loop(self):
        while True:
            value = random.choice(["True", "False"])
            attrs = {"charging": value}

            try:
                self.client.log_attributes(attrs)
                print(f"[ATTR] Logged: {attrs} (server functions only selectes clients which are charging).")
            except Exception as e:
                print(f"Failed logging attributes: {e}")

            time.sleep(30)

    def ping(self, scaleout_model: ScaleoutModel):
        pass



    def train(
        self,
        scaleout_model: ScaleoutModel,
        settings,
        data_path=None,
        batch_size=32,
        epochs=1,
        data_loader=None
    ):
        """Complete a model update.

        Load model paramters from ScaleoutModel (managed by the FEDn client),
        perform a model update, and return updated parameters wrapped in a
        ScaleoutModel together with training metadata.

        :param scaleout_model: The incoming model parameters.
        :type scaleout_model: ScaleoutModel
        :param settings: Client settings (currently unused).
        :type settings: dict
        :param data_path: The path to the data file.
        :type data_path: str
        :param batch_size: The batch size to use.
        :type batch_size: int
        :param epochs: The number of epochs to train.
        :type epochs: int
        """
        @torch.no_grad()
        def frobenius_norm(global_model, local_model):
            """
            Returns Frobenius norm of the difference between local model and global model.
            
            :param global_model: Global model from server
            :param local_model: local model after training on client data
            """
            total = 0.0

            for p_global, p_local in zip(global_model.parameters(), local_model.parameters()):
                if p_local.requires_grad:
                    total += torch.sum((p_local - p_global).float() ** 2)

            return total.detach().item()
        
        def estimate_fisher(model, loader, criterion):
            fisher = {n: torch.zeros_like(p)
                    for n, p in model.named_parameters()
                    if p.requires_grad}

            model.eval()

            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                model.zero_grad(set_to_none=True)
                output = model(x)
                loss = criterion(output, y)
                loss.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach()**2

            for n in fisher:
                fisher[n] /= len(loader)

            return fisher
        

        def get_flat_grad(model):
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().view(-1))
                return torch.cat(grads)
        
        def accumulate_gradients(model, data_loader, criterion):

            model.zero_grad(set_to_none=True)
            all_outs = []

            for x, y in data_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                outputs = model(x)
                loss = criterion(outputs, y)

                loss.backward()
                all_outs.append(outputs.detach())  # prevent graph buildup

            return get_flat_grad(model), torch.cat(all_outs, dim=0)
        
        @torch.no_grad()
        def weight_cosine_sim(global_model, local_model):
            global_weights = torch.nn.utils.parameters_to_vector(global_model.parameters())
            local_weights = torch.nn.utils.parameters_to_vector(local_model.parameters())

            sim = torch.nn.functional.cosine_similarity(local_weights, global_weights, dim=0)
            return sim.item()



        # Load model
        model = load_parameters(scaleout_model)
        model.to(self.device)
        model.train()
        lr = settings["learning_rate"]

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.NLLLoss()

        #Compute gradient for global model on local data
        global_gradient, all_outs_global = accumulate_gradients(model, data_loader, criterion)

        n_samples = len(data_loader.dataset)
        n_batches = len(data_loader)
        all_outs_local = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            #all_outs = torch.zeros_like(n_batches)
            for b, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # Regularly check if task is aborted
                self.client.check_task_abort()


                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)

                loss = criterion(outputs, y)
                loss.backward()

                all_outs_local.append(outputs.detach())
                optimizer.step()

                running_loss += loss.item() * x.size(0)

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

                if b % 100 == 0:
                    print(
                        f"Epoch {epoch}/{epochs - 1} | "
                        f"Batch {b}/{n_batches - 1} | "
                        f"Loss: {loss.item():.4f}"
                    )

            epoch_loss = running_loss / total
            epoch_acc = correct / total if total > 0 else 0.0

            # Log metrics to Scaleout
            self.client.log_metric(
                {
                    "training_loss": float(epoch_loss),
                    "training_accuracy": float(epoch_acc),
                }
            )

            #Get all metrics
            local_gradient = get_flat_grad(model)
            all_outs_local = torch.cat(all_outs_local, dim=0)

            #Get global model
            global_model = load_parameters(scaleout_model).to(self.device)
            
            norm = frobenius_norm(global_model=global_model, local_model=model)

            #fisher_diag = estimate_fisher(model, data_loader, criterion)

            #local_gradient, all_outs_local = accumulate_gradients(model, data_loader, criterion)

            feature_drift = torch.sum((all_outs_local - all_outs_global)**2).item()

            sim = weight_cosine_sim(global_model, model)

            self.metric = sim #torch.linalg.vector_norm(local_gradient - global_gradient).item()
            #Max metrics för att se lokalt
            metrics = {
                    "training_loss": float(epoch_loss),
                    "training_accuracy": float(epoch_acc),
                    "norm": norm,
                    #"fisher": fisher_diag,
                    "local_grad": local_gradient,
                    "global_grad": global_gradient,
                    "weight_cosine_sim": sim,
                    "feature_drift": feature_drift
                }
            
            print(
                f"Epoch {epoch} finished | "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
            )

        # Metadata needed for aggregation server side
        metadata = {
            "num_examples": int(n_samples),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "metric": float(self.metric),
            "training_loss": float(epoch_loss),
            "training_accuracy": float(epoch_acc),
            "norm": norm,
            "local_grad": local_gradient,
            "global_grad": global_gradient,
            "weight_cosine_sim": sim,
            "feature_drift": feature_drift
        }
        print(self.metric)
        # Save model update (mandatory)
        result_model = save_parameters(model)
        return result_model, metadata, metrics

    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Validate model.

        :param scaleout_model: The incoming model parameters.
        :type scaleout_model: ScaleoutModel
        :param data_path: The path to the data file.
        :type data_path: str
        :return: A JSON-serializable report dict.
        :rtype: dict
        """
        # Load data
        x_train, y_train = load_data(data_path)
        x_test, y_test = load_data(data_path, is_train=False)

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device).long()
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device).long()

        # Load model
        model = load_parameters(scaleout_model)
        model.to(self.device)
        model.eval()

        criterion = torch.nn.NLLLoss()

        with torch.no_grad():
            train_out = model(x_train)
            training_loss = criterion(train_out, y_train)
            training_preds = torch.argmax(train_out, dim=1)
            training_accuracy = (training_preds == y_train).float().mean()

            test_out = model(x_test)
            test_loss = criterion(test_out, y_test)
            test_preds = torch.argmax(test_out, dim=1)
            test_accuracy = (test_preds == y_test).float().mean()

        report = {
            "training_loss": float(training_loss.item()),
            "training_accuracy": float(training_accuracy.item()),
            "test_loss": float(test_loss.item()),
            "test_accuracy": float(test_accuracy.item()),
        }

        return report

    def predict(self, scaleout_model: ScaleoutModel, data_path=None):
        """Predict on test data (or other data).

        :param scaleout_model: The incoming model parameters.
        :type scaleout_model: ScaleoutModel
        :param data_path: The path to the data file.
        :type data_path: str
        :return: Dict with predictions.
        :rtype: dict
        """
        # Using test data for prediction but another dataset could be loaded
        x_test, _ = load_data(data_path, is_train=False)
        x_test = x_test.to(self.device)

        # Load model
        model = load_parameters(scaleout_model)
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            y_pred = model(x_test)
            y_pred = torch.argmax(y_pred, dim=1)

        return {"predictions": y_pred.cpu().tolist()}
