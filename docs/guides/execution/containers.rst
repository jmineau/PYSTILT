In Containers (Kubernetes)
==========================

A project can run in many containers at once, such as the pods of a
Kubernetes Job, without Slurm. Each container runs one share of the
receptors:

.. code-block:: bash

   stilt run /project --task 3/20

``--task I/N`` runs task ``I`` of ``N`` in this process: every ``N``-th
receptor of the project, starting at ``I`` (0 to ``N - 1``). Every task
splits the receptors the same way, whenever it starts. Receptors whose
results exist are skipped, so running a task again only fills in what is
missing.

``--receptors ids.txt`` runs only the receptors listed in a file, one id per
line. With ``--task`` too, the task takes its share of that list.

What every container needs
--------------------------

- PYSTILT installed (``pip install pystilt``).
- The project folder (``config.yaml`` and ``receptors.csv``), the
  meteorology, and the output directory, on a volume all the containers
  share. Results are written there, one file per simulation, so the
  containers do not need to talk to each other. An object store is not
  supported yet.
- A scratch folder for HYSPLIT. Set ``PYSTILT_COMPUTE_ROOT`` to one on fast
  local disk, such as an ``emptyDir``.

A Kubernetes indexed Job
------------------------

An indexed Job gives each pod its index in ``JOB_COMPLETION_INDEX``, which
is the task number:

.. code-block:: yaml

   apiVersion: batch/v1
   kind: Job
   metadata:
     name: stilt-my-project
   spec:
     completionMode: Indexed
     completions: 20          # N, the number of tasks
     parallelism: 20          # pods running at once
     backoffLimit: 6
     template:
       spec:
         restartPolicy: Never
         containers:
           - name: stilt
             image: my-registry/pystilt:latest   # any image with PYSTILT
             command: ["sh", "-c"]
             args: ["exec stilt run /project --task $JOB_COMPLETION_INDEX/20 --cpus 4"]
             env:
               - name: PYSTILT_COMPUTE_ROOT
                 value: /scratch
             resources:
               requests:
                 cpu: "4"
                 memory: 8Gi
             volumeMounts:
               - {name: project, mountPath: /project}
               - {name: met, mountPath: /met, readOnly: true}
               - {name: scratch, mountPath: /scratch}
         volumes:
           - name: project
             persistentVolumeClaim: {claimName: stilt-project}
           - name: met
             persistentVolumeClaim: {claimName: hrrr-arl}
           - name: scratch
             emptyDir: {}

The met ``directory`` in ``config.yaml`` must be the path inside the
container (``/met`` here). ``completions`` and the ``N`` in ``--task`` must
match. ``--cpus 4`` runs four receptors at once in each pod. ``exec`` makes
``stilt`` the process that receives Kubernetes' stop signal, so a stopped
pod stops its runs cleanly and exits with code 2.

Exit codes
----------

``stilt run`` exits with:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 0
     - Every simulation it ran is complete.
   * - 1
     - Some simulations failed. ``stilt status`` counts them by reason. A
       retry runs them again, which helps only when the cause is fixed.
   * - 2
     - Some simulations did not finish, because the run was stopped (Ctrl-C,
       or a stop signal from Kubernetes or Slurm). A retry continues where
       it stopped.

When the Job is done, ``stilt status /project`` counts the finished and
failed simulations of the whole project.
