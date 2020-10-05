import tensorflow_cloud as tfc
tfc.run(entry_point='model_training_tfcloud.ipynb',
#        chief_config=tfc.COMMON_MACHINE_CONFIGS['T4_4X'],
        requirements_txt='requirements.txt')