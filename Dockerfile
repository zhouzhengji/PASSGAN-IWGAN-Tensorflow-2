FROM tensorflow/tensorflow:2.1.0-gpu
 
#
# Maintainer of this Image
LABEL maintainer="rachelahorner.com" name="PassGAN Research"

#
# Add test PassGAN files to container
COPY PassGAN/* /home/
 
#
# Expose port 80
EXPOSE 80

#
# Shoutback
RUN echo 'PassGAN Research Container developed @Plymouth University'
 
#
# Dockerfile image command
CMD [""/bin/sh", "-c""]



