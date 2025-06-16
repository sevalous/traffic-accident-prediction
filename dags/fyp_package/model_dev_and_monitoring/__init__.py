# import logging to log any errors or info to Airflow
import logging

# model construction and training
from torch.optim import Adam
from torch import Tensor
from pytorch_lightning import Trainer

from coral_pytorch.losses import corn_loss

# model monitoring
from mlflow import set_tracking_uri, set_experiment, start_run, log_params
from mlflow.pytorch import autolog

# Redis server connection
from pickle import loads
from redis import Redis

from .classes import *

def model_dev_and_monitoring__main(logger:logging.Logger, redis_host:str, redis_port:int,
                                   mlflow_uri:str, experiment_name:str) -> None:
    """
    Function that manages the program flow of the model development stage.
    """
    try:
        # hyperparameter constants
        hyperparams = {
            'batch_size':64,
            'criterion':corn_loss,
            'max_epochs':11, # total number of epochs will be max_epochs - 1
            'n_features':12,
            'hidden_size':64,
            'num_layers':2,
            'learning_rate':0.0001,
            'num_classes':5,
            'optimiser':Adam,
            'dropout':0.2
        }

        # retrieve the preprocessed data
        with Redis(host=redis_host, port=redis_port) as redis_conn:
            X_train:Tensor = loads(redis_conn.get("X_train_preprocessed"))
            X_val:Tensor = loads(redis_conn.get("X_val_preprocessed"))
            X_test:Tensor = loads(redis_conn.get("X_test_preprocessed"))
            y_train:Tensor = loads(redis_conn.get("y_train_preprocessed"))
            y_val:Tensor = loads(redis_conn.get("y_val_preprocessed"))
            y_test:Tensor = loads(redis_conn.get("y_test_preprocessed"))

            redis_conn.flushdb() # flush the temporary datastore now finished with the data

        data_module  = RoadAccidentDataModule(X_train=X_train, X_test=X_test,
                                     X_val=X_val, y_train=y_train,
                                     y_test=y_test, y_val=y_val, batch_size=hyperparams['batch_size'])
        
        neural_network = AccidentLikelihoodNetwork(n_features=hyperparams['n_features'], hidden_size=hyperparams['hidden_size'],
                                                   output_size=hyperparams['num_classes'], batch_size=hyperparams['batch_size'],
                                                   num_layers=hyperparams['num_layers'], learning_rate=hyperparams['learning_rate'],
                                                   optimiser=hyperparams['optimiser'], criterion=hyperparams['criterion'],
                                                   dropout=hyperparams['dropout'])

        # set the connection to the MLflow server instance
        set_tracking_uri(mlflow_uri)
        # set the experiment name of the model
        set_experiment(experiment_name)
        # autolog certain evaluation metrics declared in the 'AccidentLikelihoodNetwork' module
        autolog()
        
        # used for managing the training and development of any model
        trainer = Trainer(max_epochs=hyperparams['max_epochs'], enable_progress_bar=True)

        # start a new MLflow run, so it tracks 
        with start_run() as run:
            trainer.fit(model=neural_network, datamodule=data_module)
            trainer.test(model=neural_network, datamodule=data_module)
            # run final model against testing set in 'data_module'
        
        # log the hyperparameters used for the run
        log_params(params=hyperparams, run_id=run.info.run_id)

    except Exception as e:
        logger.critical("Error occured during model development stage.\n{}".format(e), exc_info=1)
    else:
        logger.info("Model development and monitoring stage successful!")