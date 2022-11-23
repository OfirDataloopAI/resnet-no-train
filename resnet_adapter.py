from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional
import pandas as pd
import numpy as np
import dtlpy as dl
import torch.nn
import logging
import torch
import time
import copy
import tqdm
import os

from dtlpy.utilities.dataset_generators.dataset_generator import collate_torch
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

logger = logging.getLogger('resnet-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for ResNet classification',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and model entities with model code implementation
    """

    def __init__(self, model_entity=None):
        if not isinstance(model_entity, dl.Model):
            # pending fix DAT-31398
            if isinstance(model_entity, str):
                model_entity = dl.models.get(model_id=model_entity)
            if isinstance(model_entity, dict) and 'model_id' in model_entity:
                model_entity = dl.models.get(model_id=model_entity['model_id'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(ModelAdapter, self).__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_model (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = self.model_entity.configuration.get('weights_filename', 'model.pth')
        # load model arch and state
        model_path = os.path.join(local_path, weights_filename)
        logger.info("Loading a model from {}".format(local_path))
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        # How to load the label_map from loaded model
        logger.info("Loaded model from {} successfully".format(model_path))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', 'model.pth')
        torch.save(self.model, os.path.join(local_path, weights_filename))
        self.configuration['weights_filename'] = weights_filename

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

            Virtual method - need to implement

        :param batch: `np.ndarray`
        :return: `list[dl.AnnotationCollection]` each collection is per each image / item in the batch
        """
        input_size = self.configuration.get('input_size', 256)
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(input_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
            ]
        )
        img_tensors = [preprocess(img.astype('uint8')) for img in batch]
        batch_tensor = torch.stack(img_tensors).to(self.device)
        batch_output = self.model(batch_tensor)
        batch_predictions = torch.nn.functional.softmax(batch_output, dim=1)
        batch_annotations = list()
        for img_prediction in batch_predictions:
            pred_score, high_pred_index = torch.max(img_prediction, 0)
            pred_label = self.model_entity.id_to_label_map.get(int(high_pred_index.item()), 'UNKNOWN')
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=pred_label),
                           model_info={'name': self.model_entity.name,
                                       'confidence': pred_score.item(),
                                       'model_id': self.model_entity.id,
                                       'dataset_id': self.model_entity.dataset_id})
            logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
            batch_annotations.append(collection)
        return batch_annotations

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        ...


def train():
    adapter = ModelAdapter()
    model = dl.models.get(model_id='63231d9982533076a48c685d')
    adapter.load_from_model(model)
    adapter.train_model(model)


def _get_imagenet_label_json():
    import json
    with open('imagenet_labels.json', 'r') as fh:
        labels = json.load(fh)
    return list(labels.values())


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'model.pth',
                                                                 'input_size': 256},
                                          output_type=dl.AnnotationType.CLASSIFICATION,
                                          )
    module = dl.PackageModule.from_entry_point(entry_point='resnet_adapter.py')
    package = project.packages.push(package_name='resnet',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop ResNet implemented in pytorch',
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai/pytorch_adapters',
                                                            git_tag='mgmt3'),
                                    modules=[module],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                        runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.7',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    # package.metadata = {'system': {'ml': {'defaultConfiguration': {'weights_filename': 'model.pth',
    #                                                                'input_size': 256},
    #                                       'outputType': dl.AnnotationType.CLASSIFICATION,
    #                                       'tags': ['torch'], }}}
    # package = package.update()
    s = package.services.list().items[0]
    s.package_revision = package.version
    s.versions['dtlpy'] = '1.63.2'
    s.update(True)
    return package


def model_creation(package: dl.Package, resnet_ver='50'):
    # bucket = dl.buckets.create(dl.BucketType.GCS,
    #                            gcs_project_name='viewo-main',
    #                            gcs_bucket_name='model-mgmt-snapshots',
    #                            gcs_prefix='ResNet{}'.format(resnet_ver))
    # artifact = dl.LocalArtifact(path=os.getcwd())

    model = package.models.create(model_name='pretrained-resnet{}'.format(resnet_ver),
                                  description='resnset{} pretrained on imagenet'.format(resnet_ver),
                                  tags=['pretrained', 'imagenet'],
                                  dataset_id=None,
                                  scope='public',
                                  # scope='project',
                                  model_artifacts=[dl.LinkArtifact(
                                      url='https://storage.googleapis.com/model-mgmt-snapshots/ResNet50/model.pth',
                                      filename='model.pth')],
                                  status='trained',
                                  configuration={'weights_filename': 'model.pth',
                                                 'batch_size': 16,
                                                 'num_epochs': 10},
                                  project_id=project.id,
                                  labels=_get_imagenet_label_json(),
                                  )
    # artifact = model.artifacts.upload(filepath=r"C:\Users\Shabtay\Downloads\New folder\*")
    return model


if __name__ == "__main__":
    env = 'prod'
    project_name = 'DataloopModels'
    dl.setenv(env)
    project = dl.projects.get(project_name)
    # package = project.packages.get('resnet')
    # package.artifacts.list()
    # model_creation(package=package)
