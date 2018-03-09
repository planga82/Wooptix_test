Sistema de poda para la red neuronal "LeNet".

El directorio "logdir" contiene los pesos pre-entrenados de la red
así como los datos generados en cada ejecución de poda.

El programa se ocupa de descargar el dataset en caso de no encontrarlo,
por lo que, para probar una poda basta con ejecutar el comando:

python prune_procedure.py --logdir logdir/ --layer_to_prune all
