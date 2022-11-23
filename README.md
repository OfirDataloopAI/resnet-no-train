# Torch Dataloop Model Adapters

These are pytorch model adapters examples.

1. ResNet50 (resnet_adapter.py)

Full Model Management documentation [here](https://dataloop.ai/docs).  
Jupyter notebooks with examples on inference and training [here](https://github.com/dataloop-ai/dtlpy-documentation/blob/main/tutorials/model_management/use_dataloop_zoo_models/classification/chapter.ipynb)

## Deployment

Add the model to your project:

```
import json
import dtlpy as dl

project_name = 'My Model'
project = dl.projects.get(project_name=project_name)

codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/pytorch_adapters',
                          git_tag='master')

model = project.models.create(model_name='ResNet',
                              description='Dataloop ResNet implemented in pytorch',
                              output_type=dl.AnnotationType.CLASSIFICATION,
                              codebase=codebase,
                              tags=['torch'],
                              default_configuration={
                                  'weights_filename': 'model.pth',
                                  'input_size': 256,
                              },
                              default_runtime=dl.KubernetesRuntime(),
                              entry_point='resnet_adapter.py')
```

Create the pretrained snapshot (ImageNet):

```
project = dl.projects.get(project_name)

resnet_ver = '50'  # 18 etc..
bucket = dl.buckets.create(dl.BucketType.GCS,
                           gcs_project_name='viewo-main',
                           gcs_bucket_name='model-mgmt-snapshots',
                           gcs_prefix='ResNet{}'.format(resnet_ver))

# load the imagenet label mapping into the snapshot definitions

with open('imagenet_labels.json', 'r') as f:
    labels = json.load(f)

snapshot = model.snapshots.create(snapshot_name='pretrained-resnet{}'.format(resnet_ver),
                                  description='resnet{} pretrained on imagenet'.format(resnet_ver),
                                  tags=['pretrained', 'imagenet'],
                                  dataset_id=None,
                                  status='trained',
                                  configuration={'weights_filename': 'model.pth',
                                                 'id_to_label_map': labels,
                                  project_id=project.id,
                                  bucket=bucket,
                                  labels=list(labels.values())
                                  )
```

## Clone and Edit

Fork this repo to change and add your special touch into the model code.

## Contributions

Help us get better! We welcome any contribution and suggestion to this repo.   
Open an issue for bug/features requests.
 
