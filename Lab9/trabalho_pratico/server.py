import flwr as fl
import flwr 
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy  import FedAvg
from flwr.server.strategy.aggregate import aggregate
import random
class Servidor(FedAvg):
    def __init__(self, num_clients, dirichlet_alpha, fraction_fit=0.2, fraction_eval=0.2):
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.clients_fit_ids = set()

        super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, fraction_evaluate=fraction_eval)

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""

        config = {
            'server_round': server_round,
        } 
        fit_ins = FitIns(parameters, config)

        all_clients = list(client_manager.all().values())
        num_available_clients = len(all_clients)
        sample_size = max(int(self.fraction_fit * num_available_clients), 1)

        clients_fit = random.sample(all_clients, min(sample_size, num_available_clients))

        self.clients_fit_ids = set([client.cid for client in clients_fit])

        print("Clients selected for fit:", [client.cid for client in clients_fit])


        # Return client/config pairs
        return [(client, fit_ins) for client in clients_fit]
    
    def aggregate_fit(self, server_round, results, failures):       
        parameters_list = []
        for _, fit_res in results:
            parameters = parameters_to_ndarrays(fit_res.parameters)
            exemplos   = int(fit_res.num_examples)

            parameters_list.append([parameters, exemplos])

        agg_parameters = aggregate(parameters_list)
        agg_parameters = ndarrays_to_parameters(agg_parameters)

        return agg_parameters, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {
            'server_round': server_round,
        } 

        evaluate_ins = EvaluateIns(parameters, config)

        all_clients = list(client_manager.all().values())
        num_available_clients = len(all_clients)
        sample_size = max(int(self.fraction_evaluate * num_available_clients), 1)
        clients_evaluate = [client for client in all_clients if client.cid not in self.clients_fit_ids]

        if not clients_evaluate:
            clients_evaluate = all_clients

        actual_sample_size = min(sample_size, len(clients_evaluate))

        clients_evaluate = random.sample(clients_evaluate, actual_sample_size)

        print("Clients selected for evaluation:", [client.cid for client in clients_evaluate])
        return [(client, evaluate_ins) for client in clients_evaluate]
    
    

    def aggregate_evaluate(self, server_round, results, failures):
        accuracies = []

        for _, response in results:
            acc = response.metrics['accuracy']
            accuracies.append(acc)

        avg_acc = sum(accuracies)/len(accuracies)
        print(f"Rodada {server_round} acurácia agregada: {avg_acc}")

        return avg_acc, {}