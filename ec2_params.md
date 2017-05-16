Connect to AWS instance :
ssh -i ~/.ssh/aws_north_california.pem ubuntu@ec2-35-156-209-66xxxxxxxxx

Copy file to instance :
 scp -i ~/.ssh/aws_north_california.pem  file_path ubuntu@ec2-35-156-209-66xxxxxxxxx:~

Connect to jupyter :
https://ec2-54-193-55-236.us-west-1.compute.amazonaws.com:8888


when restarting instance, ssh -i ~/.ssh/aws_north_california.pem ubuntu@ec2-35-156-209-66xxxxxxxxx


then in the root folder :
sudo mount /dev/xvdba /drive_ext4


do not forget to stop spot request and instance itself.


## Amazon AMI Deep Learning

Those are Amazon Linux machine

- ssh -i ~/.ssh/name_of_key.pem  ec2-user@ec2-xxxxxx.us-west-2.compute.amazonaws.com
- check if GPU works with tensore : in /home/ec2-user/src/bin run :`./testTensorFlow`
- look at `nvidia-smi` and check that GPU as a python process running.
- run the code



