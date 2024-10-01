import boto3

def monitor_ec2_instances():
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_instances()
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            instance_state = instance['State']['Name']
            print(f"Instance ID: {instance_id}, State: {instance_state}")

monitor_ec2_instances()