import mlflow
import time
import subprocess
from cityscape_generator import get_split_generator
from data_tools import visualize_model_prediction
from model_unet import Unet
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def start_local_experiment( host='127.0.0.1',
                            port='8080',
                            uri=r'/mlruns',
                            experiment_name="cityscape"
                            ):
    command = f'''mlflow server --host {host}  --port {port} \n
                mlflow ui --backend-store-uri {uri}'''
    print(command)

    result = subprocess.Popen(command, shell=True)

    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

    mlflow.set_experiment(experiment_name)



def cityscape_experiment(model_func = Unet,
                         model_params = {
                             'img_height' : 256,
                             'img_width'  : 256,
                             'nclasses'   : 8
                         },
                         gen_params = {'batch_size':4},
                        metrics = [
                                    # dice_coeff,
                                    'accuracy'],
                        train_step_per_epoch = 93, #sample/batch_size
                        val_step_per_epoch = 16,
                        num_epochs = 8,
                        training_description = 'cityscape segmentation experiments',
                        model_name = 'Unet',
                        
                        **kwargs

                        ):
    start_time = time.time()

    model = model_func(**model_params)

    generator_dic = get_split_generator(params=gen_params)
    train_generator = generator_dic['train']

    if train_step_per_epoch is None:
        train_step_per_epoch = int(len(train_generator.image_list)/train_generator.batch_size)
    
    val_generator= generator_dic['val']

    if val_step_per_epoch is None:
        val_step_per_epoch = int(len(val_generator.image_list)/val_generator.batch_size)

    optimizer=kwargs.get('optimizer','adam')
    loss = kwargs.get('loss','categorical_crossentropy')
    monitor=kwargs.get('monitor','val_loss')
    model.compile(optimizer=optimizer, 
                  loss=loss, metrics=metrics)
    
    
    mlflow.tensorflow.autolog()
    callbacks = []
    if kwargs.get('use_tensorboard',False):
        tb = TensorBoard(log_dir='logs', write_graph=True)
        callbacks.append(tb)
    if kwargs.get('use_checkpoint',False):
        mc = ModelCheckpoint(
                            mode='max', 
                            filepath='models-dr/pdilated.weights.h5', 
                            monitor=monitor, 
                            save_best_only='True', 
                            save_weights_only='True', 
                            verbose=1)
        callbacks.append(mc)
    es = EarlyStopping(
                        monitor=monitor,
                        mode='min',
                    #    mode='max', 
                    #    monitor='acc', 
                        patience=1, 
                        verbose=1)
    callbacks.append(es)
    # vis = visualize()


    mlflow.tensorflow.autolog()
    history =  model.fit( #fit_generator deprecated
                train_generator,
                steps_per_epoch=train_step_per_epoch,
                epochs=num_epochs,
                verbose=1,
                validation_data=val_generator,
                validation_steps=val_step_per_epoch,
                # use_multiprocessing=True,
                # workers=16,
                callbacks=callbacks,
                # max_queue_size=32,
                )

    process_time = time.time() - start_time

    prediction_test_fig = visualize_model_prediction(model)

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.log_figure(prediction_test_fig, "prediction_test.png")

        signature = None
        # Infer the model signature
        # if sign_model:
        #     signature = infer_signature(X_train, model.predict(X_train), model_params )


        model_info  = mlflow.keras.log_model(
                                    model=model,        
                                    name=model_name,
                                    signature=signature,
                                    input_example=None,
                                    registered_model_name=f"{model_name}",
                                    )

        
        # hash_id = None
        # try:
        #     import hashlib
        #     hash_id = hashlib.sha256(df.to_string().encode()).hexdigest()
        # except:
        #     pass

        # Log other information about the model
        mlflow.log_params({ "Process_Time": process_time,
                           'ModelParams' : model_params,
                            'optimizer':optimizer,
                            'loss':loss,
                            'monitor':monitor,
                            'GenParams' : gen_params,
                            'Metrics' : metrics,
                            'TrainStepPerEpoch' : train_step_per_epoch, #sample/batch_size
                            'ValStepPerEpoch' : val_step_per_epoch,
                            'Epochs' : num_epochs,
                            # 'DataHash': hash_id,
                            # 'dataset_path':df_path,
                            # 'dataset_length':len(df.index),
                            
                            })

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": training_description,
                                  
                                  }
        )

        # mlflow.keras.load_model(model_info.model_uri)