# -*- coding: utf-8 -*-

import time
import gzip
import logging
import argparse
import datetime
from google.cloud import pubsub

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TOPIC = 'sandiego'
PROJECT = 'woven-rush-197905'


def create_subscription(project, topic_name, subscription_name):
    """Create a new pull subscription on the given topic."""
    subscriber = pubsub.SubscriberClient()
    topic_path = subscriber.topic_path(project, topic_name)
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    subscription = subscriber.create_subscription(
        subscription_path, topic_path)

    print('Subscription created: {}'.format(subscription))


def create_push_subscription(project,
                             topic_name,
                             subscription_name,
                             endpoint):
    """Create a new push subscription on the given topic.
    For example, endpoint is
    "https://my-test-project.appspot.com/push".
    """
    subscriber = pubsub.SubscriberClient()
    topic_path = subscriber.topic_path(project, topic_name)
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    push_config = pubsub.types.PushConfig(
        push_endpoint=endpoint)

    subscription = subscriber.create_subscription(
        subscription_path, topic_path, push_config)

    print('Push subscription created: {}'.format(subscription))
    print('Endpoint for subscription is: {}'.format(endpoint))


def list_subscriptions_in_topic(project, topic_name):
    """Lists all subscriptions for a given topic."""
    subscriber = pubsub.PublisherClient()
    topic_path = subscriber.topic_path(project, topic_name)

    for subscription in subscriber.list_topic_subscriptions(topic_path):
        print(subscription)


def delete_subscription(project, subscription_name):
    """Deletes an existing Pub/Sub topic."""
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    subscriber.delete_subscription(subscription_path)

    print('Subscription deleted: {}'.format(subscription_path))


def receive_messages(project, subscription_name):
    """Receives messages from a pull subscription."""
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    def callback(message):
        print('Received message: {}'.format(message))
        message.ack()

    subscription = subscriber.subscribe(subscription_path, callback=callback)

    # Blocks the thread while messages are coming in through the stream. Any
    # exceptions that crop up on the thread will be set on the future.
    try:
        subscription.future.result()
    except Exception as e:
        print('Listening for messages on {} threw an Exception: {}.'.format(
            subscription_name, e))
        raise


if __name__ == '__main__':
    # list_subscriptions_in_topic(PROJECT, TOPIC)
    # create_subscription(PROJECT, TOPIC, 'test_sub')
    # list_subscriptions_in_topic(PROJECT, TOPIC)
    receive_messages(PROJECT, 'test_sub')
