"""An isolated copy of the task's official compose services per agent session."""
import json
import subprocess
from pathlib import Path

from benchmark.images import source_label
from benchmark.terminal_task import check_source


# Build the official service Dockerfiles unchanged; each session gets its own
# project, containers and network. These are task environments, not agent peers.
SERVICE_BUILDS = {
    'nextjs-performance': {'warehouse-api': ('.', 'Dockerfile.api')},
    'legacy-utility-triage': {
        'legacy-workstation': ('workstation', 'Dockerfile'),
        'legacy-app': ('legacy_app', 'Dockerfile'),
    },
    'payments-pipeline-fix': {
        'seeder': ('seeder', 'Dockerfile'),
        'customer': ('customer', 'Dockerfile'),
    },
    'cumulative-layout-shift': {'barber-shop-data-backend': ('barber-shop-data-backend', 'Dockerfile')},
    'kv-live-surgery': {'loadgen': ('loadgen', 'Dockerfile')},
    'live-database-cutover': {
        'mysql-db': ('mysql', 'Dockerfile'),
        'postgres-db': ('postgres', 'Dockerfile'),
        'customer': ('customer', 'Dockerfile'),
    },
}

SERVICE_IMAGES = {
    'payments-pipeline-fix': {'kafka': 'apache/kafka-native:4.3.1'},
    'live-database-cutover': {'redis': 'redis:7-alpine'},
}


def service_images(task_name):
    return {**{service: f'delm-sidecar-{task_name}-{service}:local'
               for service in SERVICE_BUILDS.get(task_name, {})},
            **SERVICE_IMAGES.get(task_name, {})}


def build_services(task, folder, *, source=None):
    folder.mkdir(parents=True, exist_ok=True)
    images = service_images(task.name)
    label = source_label(check_source(task) if source is None else source)
    for service, (context, dockerfile) in SERVICE_BUILDS.get(task.name, {}).items():
        image = images[service]
        root = task / 'environment' / context
        with (folder / f'build-{service}.log').open('w') as log:
            subprocess.run(['docker', 'build', '--provenance=false', '--label', label, '-f', str(root / dockerfile), '-t', image, str(root)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    for service, image in SERVICE_IMAGES.get(task.name, {}).items():
        if subprocess.run(['docker', 'image', 'inspect', image], capture_output=True).returncode == 0:
            continue
        with (folder / f'pull-{service}.log').open('w') as log:
            subprocess.run(['docker', 'pull', image], check=True, stdout=log, stderr=subprocess.STDOUT)
    return images


class TaskCompose:
    def __init__(self, name, image, folder, limits, environment, mounts):
        self.folder = folder
        task_name = Path(limits['compose_file']).parent.parent.name
        self.services = limits.get("service_image_ids", service_images(task_name))
        if not self.services:
            raise ValueError(f'Unsupported compose task: {task_name}')
        sidecars = {service: name + '-' + service for service in self.services}
        # CLS runs browsers alongside Next.js; its official task sets no PID cap.
        pids_limit = {'cumulative-layout-shift': -1, 'nextjs-performance': 2048}.get(task_name, 512)
        override = folder / 'compose.override.json'
        services = {
            'main': dict(image=image, container_name=name, command=['sleep', 'infinity'],
                         cpus=limits['environment']['cpus'],
                         extra_hosts=['host.docker.internal:host-gateway'],
                         mem_limit=f"{limits['environment']['memory_mb']}m",
                         pids_limit=pids_limit,
                         labels={'delm.workspace.research': 'true'},
                         environment={k: str(v) for k, v in environment.items()},
                         volumes=[dict(type='bind', source=str(source), target=target, read_only=True)
                                  for source, target in mounts] + [
                             dict(type='bind', source=str(folder), target='/records')])}
        services.update({service: dict(image=image,
                                  container_name=sidecars[service],
                                  extra_hosts=['host.docker.internal:host-gateway'],
                                  labels={'delm.workspace.research': 'true'})
                         for service, image in self.services.items()})
        override.write_text(json.dumps(dict(services=services), indent=2))
        self.command = ['docker', 'compose', '-p', name, '-f', limits['compose_file'], '-f', str(override)]
        try:
            with (folder / 'compose-start.log').open('w') as log:
                subprocess.run(self.command + ['up', '-d', '--no-build', '--pull', 'never',
                                               '--wait', '--wait-timeout', '150'],
                               check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            containers = json.loads(subprocess.check_output(['docker', 'inspect', name, *sidecars.values()]))
            (folder / 'compose-services.json').write_text(json.dumps([
                dict(name=c['Name'], image=c['Image'], network_mode=c['HostConfig']['NetworkMode'],
                     networks=c['NetworkSettings']['Networks']) for c in containers], indent=2))
        except BaseException:
            self.stop()
            raise

    def stop(self):
        with (self.folder / 'compose-services.log').open('w') as log:
            subprocess.run(self.command + ['logs', '--no-color', *self.services],
                           stdout=log, stderr=subprocess.STDOUT, timeout=30)
        subprocess.run(self.command + ['stop', '-t', '5'], capture_output=True, timeout=30)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build official task service images without running agents.')
    parser.add_argument('task', type=Path)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_services(args.task.resolve(), args.folder.resolve())))
