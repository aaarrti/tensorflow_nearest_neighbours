# Build base image
apptainer build --fakeroot --nv container.sif container.def
# Create image for venv
mkfs.ext3 -E root_owner venv.img 10G
# Create venv inside of image
apptainer exec --nv -B venv.img:/venv:image-src=/ container.sif /usr/bin/python3 -m venv /venv
# Copy tensorflow installation into venv
apptainer exec --nv -B venv.img:/venv:image-src=/ container.sif cp -r /usr/local/lib/python3.8/dist-packages /venv/lib/python3.8/site-packages
# Run shell in container with venv bounded at root
apptainer shell --nv -B venv.img:/venv:image-src=/,ro container.sif
