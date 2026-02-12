pre: sök och fyll i formuläret och skicka för imagenet1k

1. python3 -m venv .venv
2. source .venv/bin/activate
## får scaleout-util/ och scaleout-client-python/ 
## från viktor
3. cd scaleout-util
4. pip install -e .
5. cd ../scaleout-client-oython
6. pip install -e .
## färdig installerat.
7. se scaleout-client-python/examples/server-functions/...
8. se och förstå server_functions.py
9. se och förstår client/startup.py
10. bygg en simulation engine med client + server_functions.py
11. gör en ny exempel folder

model = ...
använd server_functions för att välja clients
använd server_functions "client_settings" funktion för att hämta ev. inställningar till klienterna.
skicka model + settings till client som selectats
samla resultat [...]
skicka resultat till server_functions för aggregering

### för varje runda (modellhantering): 

initial_model = deepcopy(model.state_dict())
loop:
    model.set_state_dict(initial_model) # reset model
    send model to client n who trains on model
    client_n_weights = deepcopy(model.state_dict()) # gör om denna till lista och sen gör deep copy
