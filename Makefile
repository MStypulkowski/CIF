project_name := CIF
container_ssh_port :=  # TODO: Pick a port for SSH
container_jupyter_port :=  # TODO: Pick a port for Jupyter
container_tensorflow_port :=  # TODO: Pick a port for Tensorboard

username := $(shell whoami)
uid := $(shell id -u)
gid := $(shell id -g)


src_dir := "/cluster-home/${username}/CIF"
results_dir := "/cluster-home/${username}/results" # TODO: Set correct path
data_dir := "/cluster-home/${username}/data" # TODO: Set correct path

docker_image_tag := "$(shell echo "${project_name}" | tr '[:upper:]' '[:lower:]')"
docker_container_name := "${username}-${docker_image_tag}"

all: build run

build:
	docker build \
		--tag ${docker_image_tag} \
		--file Dockerfile  \
		--build-arg username=${username} \
		--build-arg uid=${uid} \
		--build-arg gid=${gid} \
		.

run:
	if [ ! -d "${results_dir}" ]; then mkdir "${results_dir}"; fi
	if [ ! -d "${data_dir}" ]; then mkdir "${data_dir}"; fi

	docker run \
		--detach \
		--name="${docker_container_name}" \
		--interactive \
		--tty \
		--publish ${container_ssh_port}:4444 \
		--publish ${container_tensorflow_port}:6006 \
		--publish ${container_jupyter_port}:8888 \
		--volume ${src_dir}:/src \
		--volume ${results_dir}:/results \
		--volume ${data_dir}:/data \
		${docker_image_tag}:latest

rm:
	docker stop ${docker_container_name}
	docker rm ${docker_container_name}
