from scaleoututil.serverfunctions.allowed_import import Dict, List, ServerFunctionsBase, Tuple, api_client, np
from scaleoututil.serverfunctions.serverfunctionsbase import RoundType

# See allowed_imports for what packages you can use in this class.


# Class must be named ServerFunctions
class ServerFunctions(ServerFunctionsBase):
    # toy example to highlight functionality of ServerFunctions.
    def __init__(self) -> None:
        # You can keep a state between different functions to have them work together.
        self.round = 0
        self.lr = 0.1

    # Skip any function to use the default FEDn implementation for the function.

    # Called first in the beggining of a round to select clients.
    def client_selection(self, client_ids: list[str], round_type: RoundType) -> list[str]:
        """Select clients that are currently charging.
        If no attributes exist (empty response) or service error, default to selecting all.
        """
        # chose every client for predictions
        if round_type == RoundType.PREDICTION:
            return client_ids
        # for train and evaluation only chose clients which are in "charging" state
        try:
            attrs_map = api_client.get_current_attributes(client_ids)
        except Exception as e:
            print(f"Warning: unable to fetch attributes ({e}), selecting all clients")
            return client_ids

        selected = []
        charging_count = 0
        for cid in client_ids:
            client_attrs = attrs_map.get(cid) or {}
            charging = client_attrs.get("charging", None)
            print(f"Client {cid} is charging: {charging}")
            # Default to select to not depend on beta version of attributes.
            if charging == "True" or charging is None:
                selected.append(cid)
                charging_count += 1
        if len(selected) < 20:
            print(f"Selected clients: {selected}.")
        else:
            print(f"Selected {len(selected)} clients.")
        print(f"{charging_count} clients selected based on client attributes, out of {len(client_ids)} clients.")
        return selected
    
    def adaptive_client_selection(self, client_ids: list[str], round_type: RoundType, clients) -> list[str]:
        if round_type == RoundType.PREDICTION:
            return client_ids
        
        metrics = [clients[id].metric for id in client_ids]
        if None in metrics:
            print("No client metrics found, selecting all clients.")
            return client_ids
        selected = np.argsort(metrics)

        return selected[:3] #Sort by metrics, lowest first

    # Called secondly before sending the global model.
    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        # Decrease learning rate every 10 rounds
        if self.round % 10 == 0:
            self.lr = self.lr * 0.1
        # see client/train.py for how to load the client settings.
        self.round += 1
        return {"learning_rate": self.lr}

    # Called third to aggregate the client updates.
    def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # Weighted fedavg implementation.
        if len(client_updates) == 0:
            print("Received no client updates. Returning previous model.")
            return previous_global
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = 0
        for client_id, (client_parameters, metadata) in client_updates.items():
            num_examples = metadata.get("num_examples", 1)
            total_weight += num_examples
            for i in range(len(weighted_sum)):
                weighted_sum[i] += client_parameters[i] * num_examples

        print("Models aggregated")
        averaged_updates = [weighted / total_weight for weighted in weighted_sum]
        return averaged_updates
    
    # def adaptive_aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]], threshold, choice) -> List[np.ndarray]:
    #     import torch
    #     weighted_sum = [np.zeros_like(param) for param in previous_global]
    #     total_weight = 0
    #     for client_id, (client_parameters, metadata) in client_updates.items():
            
    #         if choice == 'grad-diff':
    #             g_global = metadata.get('global_grad')
    #             g_local = metadata.get('local_grad')

    #             norm = torch.linalg.vector_norm(g_local - g_global, dim=0).item()

    #             total_weight += norm

    #             for i in range(len(weighted_sum)):
    #                 weighted_sum[i] += client_parameters[i] * norm


    #         if choice == 'cosine-sim':

    #             total_weight += (1 - metadata.get("metric", 0) )# use metric as weight instead of num examples

    #             for i in range(len(weighted_sum)):
    #                 weighted_sum[i] += client_parameters[i] * (1 - metadata.get("metric", 0))


    #     print("Models aggregated")
    #     averaged_updates = [weighted / total_weight for weighted in weighted_sum]
    #     return averaged_updates


    def adaptive_aggregate(self, previous_global, client_updates, threshold, choice):
        import torch
        weighted_sum = [np.zeros_like(p) for p in previous_global]
        total_weight = 0.0

        for client_id, (client_parameters, metadata) in client_updates.items():

            if choice == 'grad-diff':
                g_global = metadata['global_grad']
                g_local = metadata['local_grad']

                weight = torch.linalg.vector_norm(g_local - g_global)
                weight = weight.detach()

            elif choice == 'cosine-sim':
                metric = metadata.get("metric", 0)
                weight = 1 - metric

            else:
                raise ValueError("Unknown choice")

            total_weight += weight

            for i, param in enumerate(client_parameters):
                weighted_sum[i] += param * weight

        averaged = [w / total_weight for w in weighted_sum]
        return averaged
