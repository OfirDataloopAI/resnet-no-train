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


class ModelAdapter(dl.BaseModelAdapter):
    """
    resnet Model adapter using pytorch.
    The class bind Dataloop model and snapshot entities with model code implementation
    """

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            Virtual method - need to implement

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = self.snapshot.configuration.get('weights_filename', 'model.pth')
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

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

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
            pred_label = self.snapshot.id_to_label_map.get(int(high_pred_index.item()), 'UNKNOWN')
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=pred_label),
                           model_info={'name': self.model_entity.name,
                                       'confidence': pred_score.item(),
                                       'model_id': self.model_entity.id,
                                       'snapshot_id': self.snapshot.id})
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


def _get_imagenet_label_json():
    import json
    with open('imagenet_labels.json', 'r') as fh:
        labels = json.load(fh)
    return list(labels.values())


def model_creation(project_name, env: str = 'prod'):
    dl.setenv(env)
    project = dl.projects.get(project_name)

    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/pytorch_adapters',
                              git_tag='master')
    model = project.models.create(model_name='ResNet',
                                  description='Global Dataloop ResNet implemeted in pytorch',
                                  output_type=dl.AnnotationType.CLASSIFICATION,
                                  scope='public',
                                  codebase=codebase,
                                  tags=['torch'],
                                  default_configuration={
                                      'weights_filename': 'model.pth',
                                      'input_size': 256,
                                  },
                                  default_runtime=dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                       runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.6',
                                                                       autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                           min_replicas=0,
                                                                           max_replicas=1),
                                                                       concurrency=1),
                                  entry_point='resnet_adapter.py')
    return model


def snapshot_creation(project_name, model: dl.Model, env: str = 'prod', resnet_ver='50'):
    dl.setenv(env)
    project = dl.projects.get(project_name)
    bucket = dl.buckets.create(dl.BucketType.GCS,
                               gcs_project_name='viewo-main',
                               gcs_bucket_name='model-mgmt-snapshots',
                               gcs_prefix='ResNet{}'.format(resnet_ver))
    snapshot = model.snapshots.create(snapshot_name='pretrained-resnet{}'.format(resnet_ver),
                                      description='resnset{} pretrained on imagenet'.format(resnet_ver),
                                      tags=['pretrained', 'imagenet'],
                                      dataset_id=None,
                                      scope='public',
                                      # status='trained',
                                      configuration={'weights_filename': 'model.pth',
                                                     'classes_filename': 'classes.json'},
                                      project_id=project.id,
                                      bucket=bucket,
                                      labels=_get_imagenet_label_json()
                                      )
    return snapshot


def model_and_snapshot_creation(project_name, env: str = 'prod', resnet_ver='50'):
    model = model_creation(project_name, env=env)
    print("Model : {} - {} created".format(model.name, model.id))
    snapshot = snapshot_creation(project_name, model=model, env=env, resnet_ver=resnet_ver)
    print("Snapshot : {} - {} created".format(snapshot.name, snapshot.id))
