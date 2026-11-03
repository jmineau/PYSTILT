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
  containers do not need to talk to each other. The output directory can
  instead be on an object store (``output: s3://bucket/...``, see
  :doc:`projects`), with the containers given its credentials.
  ``stilt run --output s3://bucket/...`` uses another output directory
  than ``config.yaml`` names, without editing it.
- A compute root, where each run gets its workdir. Set
  ``PYSTILT_COMPUTE_ROOT`` to a folder on fast local disk, such as an
  ``emptyDir``.

.. _fetch-once:

Fetching the meteorology once
-----------------------------

With ``download`` set, a run downloads the met files its hours need into the
met ``directory``. Pods on one shared volume find the files the first pod
fetched. Pods with their own disks do not: 50 pods whose receptors fall on
the same days download, and crop, the same files 50 times.

``stilt fetch`` gets the files once, before the Job starts:

.. code-block:: bash

   stilt fetch /project

For each met it gets the files that cover the time of every simulation that
has yet to run, cropped to the met's subgrid. They are the files ``stilt
run`` would ask for, so a run that follows finds them all and downloads
nothing. Run it on one pod, with the met ``directory`` on the volume the
Job's pods then mount. Simulations that are finished need no met and are
left out; ``--output`` is as for ``stilt run``, so it reads the same
results.

A met without ``download`` is only checked: ``stilt fetch`` reports the
files it lacks, as ``stilt check`` does, and does not make them.

It exits with 0 when every file is there, 1 when some could not be had
(the archive does not have the file; it prints the times of the runs that
need them), 2 when the command line is wrong or a met directory cannot be
read, and 3 when it was stopped (Ctrl-C, or SIGTERM, which Kubernetes and
Slurm send) or the archive was busy or the connection failed (a throttled
bucket, a timeout, a refused connection), which a retry may get past. Any other
unexpected error is 4 (:ref:`exit-codes`). A stop removes the file it was downloading,
and a second stop signal while it does is ignored. Run it again to try the
missing files, or to finish after a stop: the ones it got are kept. A process
killed with SIGKILL, which cannot be caught, leaves its hidden
``.<name>.<random>.partial`` file in the met directory; delete it.

Fetches can overlap. Several ``stilt fetch`` processes, or a fetch beside
runs, can share one met directory: each file is written under a temporary
name and renamed into place when it is whole, so no one reads half a file.
Two that start the same file both download it, and the second to finish
replaces the first with the same contents.

In Python, ``project.fetch()`` does the same and returns the files, one row
each, with the times it could not get files for.

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
pod stops its runs cleanly and exits with code 3, within a few seconds of
the signal and so inside Kubernetes' default 30 s grace period. That holds
for ``--cpus`` above 1 too: the pod's workers and HYSPLIT runs stop with it.

.. _exit-codes:

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
     - The command line was wrong: an unknown option, a bad value, or a
       directory that is not a project. Or a met directory does not exist
       or holds no ARL file (``stilt check`` lists it). Nothing ran. Do not
       retry.
   * - 3
     - Retry it. Some simulations did not finish, because the run was
       stopped (Ctrl-C, or a stop signal from Kubernetes or Slurm), or a
       connection or I/O error, such as an output bucket that could not be
       reached, stopped an unfinished simulation or ended the run. A retry
       continues where it stopped.
   * - 4
     - A bug: any other error the command did not expect. Its traceback is
       printed. Do not retry; a retry meets the same error.

Whether an error is a connection or I/O error is read from the error and
what it was raised from, not from its being an ``OSError``: s3fs turns many
permanent answers of a bucket into a plain ``OSError``. It is 3 when it is a
refused, reset or timed-out connection or a name-resolution failure, an error of
the cloud libraries that is not an ``OSError`` (such as botocore's
``EndpointConnectionError`` and ``ReadTimeoutError``), an FTP 4xx reply ("421
too many users"), or has the status of a busy or briefly unavailable service
(408, 429, 500, 502, 503, 504, and the throttling codes such as ``SlowDown``).
When an error hides its cause, as fsspec's HTTP filesystem does by raising
``FileNotFoundError`` for every failure, the cause decides. An ``OSError`` that nothing explains is
3 when its errno is input/output trouble that goes away (``EIO``, a stale
NFS handle, a busy or interrupted call) and 4 otherwise, a full disk
(``ENOSPC``) included. It is 4 when the status is any other 3xx, 4xx or 5xx (a
bucket in another region, a bad request, a 403, a 404), an FTP 5xx reply, a file
that is not there or may not be used, or anything else. Every ``stilt`` command ends this way on
an error it did not expect, not only ``stilt run``, except that a closed
standard output (``stilt status | head``) ends quietly with 1, as it always
has. ``stilt fetch`` follows the same table, except that
1 is a file the archive does not have (a 404, whatever library reports it).

The same rule applies to an error inside one simulation, such as a worker
that cannot write its particles or footprint to the output bucket. An error
that is 3 by the rule above is not a failure of the simulation: no failure
record is written (an earlier attempt's is removed), the simulation stays
unfinished (``pending``, or ``interrupted`` once its particles run had started),
the other simulations go on, and the run ends with 3 so that a retry picks it
up. Any other error inside a simulation, a ``SimulationError`` such as a
failed HYSPLIT run or missing met files included, a bucket answer that a retry
repeats (access denied, no such key), or a full disk, is a failure record and
exit 1. When both happen in one run it exits 3, and the retry, which runs only
what did not finish, ends with 1 if the failure remains. A bucket that stays
down keeps every try at 3, so a driver sets how many times it retries. If the
output cannot be reached to remove an earlier failure record, that record stays
until a run writes the result.

For a Kubernetes indexed Job, a ``podFailurePolicy`` that retries on 3 and
fails the task on 1 and 4 follows from this table.

When the Job is done, ``stilt status /project`` counts the finished and
failed simulations of the whole project. ``stilt status /project --json``
gives the same counts, per state and variant, as JSON for the program
that drives the Job.
