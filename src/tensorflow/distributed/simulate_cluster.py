# -*- coding: utf-8 -*-

import tensorflow as tf

cluster_spec_dict = {
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225"]
}
spec = tf.train.ClusterSpec(cluster_spec_dict)
servers = []


def createServers():
    for job in spec.jobs:
        for idx, task in enumerate(spec.job_tasks(job_name=job)):
            server = tf.train.Server(spec, job_name=job, task_index=idx)
            servers.append(server)


# import os
# import json
#
# tf_config = os.environ.get('TF_CONFIG')  # If TF_CONFIG is not available run local
# if not tf_config:
#     return run('', True, *args, **kwargs)
# tf_config_json = json.loads(tf_config)
# cluster = tf_config_json.get('cluster')
# cluster_spec = tf.train.ClusterSpec(cluster)
