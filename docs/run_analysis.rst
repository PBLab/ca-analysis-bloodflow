-----------------------------
Running the Analysis Pipeline
-----------------------------

As you may already know, this repository contains functions desgined to analyze calcium and vascular data taken by our lab. However, the first step of analysis is done using CaImAn, a Python package that analyzes calcium recording from one- and two-photon images. All of the scripts in the repo assume that CaImAn was run on the files you wish to analyze, and thus they'll process those results. Below we'll detail the  necessary steps to take in order to analyze your data using CaImAn.

CaImAn Analysis Pipeline
------------------------

Since CaImAn is a resource hog, we (almost) never run it on our own server ("Cortex", located at ``cortex.tau.ac.il``). Instead, we'll use Cortex to look at the data, obtain a few key parameters that CaImAn needs in order to run well, and then run the actual code on the power8 server (``powerlogin.tau.ac.il``). 

1. Open a VNC session to ``cortex.tau.ac.il`` and connect to it.
2. Move all the data you wish to analyze to a single folder. 
3. Using Fiji, open one of the recordings and write down somewhere the following parameters:
   a. Radius of the cell in pixels
   b. Average number of cells per 50x50 pixels patch
   c. Frame rate
   d. The data channel (if you recorded GCaMP in channel 1 and Texas Red in 2, then the data channel is 1)
4. **Copy** the data folder to the following folder: /mnt/pblabfs/pblab/<YOUR NAME>. You can also move it to a sub-folder inside it. This means that if your data folder is called ``june12-gcamp6-mouse1`` then the final data folder is ``/mnt/pblabfs/pblab/<YOUR NAME>/june12-gcamp6-mouse1``.
5. Open and edit the file ``/mnt/pblabfs/pblab/caiman_scripts/multifile_pipeline.py``. Double-clicking it should open it in edit mode. At the end of the file, after ``if __name__ == '__main__':`` (line ~74) you'll find all of the parameters that CaImAn requires to run. Most of them should be left untouched, but a few of them are crucial.
   a. Change ``fr`` to your frame rate. It can be a non-integer number.
   b. Change ``K`` to the number of cells per 50x50 pixel patch. This should be an integer.
   c. Change ``gSig`` to the radius, in pixels, of a cell. Integer.
   d. Change ``foldername`` to the name of the folder that the data was copied to. Take note: you copied the data to ``/mnt/pblabfs/pblab/<YOUR NAME>``, but here that same path should be written as ``/pblab/pblab/<YOUR NAME>``. This is because this script will be run in a different computer, where the path is named differently. If you fail to write down the foldername properly CaImAn will simply not run, so make sure you got this one right.
   e. Change ``glob_str`` to a `glob string <https://en.wikipedia.org/wiki/Glob_%28programming%29>`_ that will capture the data you wish to analyze. If you want to analyze all files in the directory then simply write ``glob_str = '*'``. This is useful to only analyze some subset of files in a given directory. For example, if the directory contains data captured at two different magnifications, these two datasets should be analyzed differently as their FPS and cell sizes will differ. The ``glob_str`` parameter let's you do this differentiation easily.
   f. Change ``num_of_channels`` to the total number of recorded channel in that recording. Integer.
   g. Change ``data_channel`` to the data channel itself. Integer.
6. Verify that everything looks right, save the file and close it.
7. Either inside the VNC session or on your own computer connect to the power8 server by writing ``ssh -X <Moodle Username>@powerlogin.tau.ac.il``. Enter your Moodle password. Note: You cannot connect without a TAU username and password.
8. Write the following line in the terminal: ``qsub -q pablo /pblab/pblab/caiman_scripts/run_caiman.sh``. After a second or two a response should come up looking like ``XXXX.power8.tau.ac.il``, where ``XXXX`` is some number representing the job number you automatically received. You can periodically check that the job is running by typing ``qstat XXXX``, or you can simply look in the folder of the data and see new files being generated. 
9. If everything went well each file should now be accompanied by a few more files:
   a. A file containing only the data channel, its name ending with ``*CHANNEL_X.tif``.
   b. A file containing the motion-corrected data, its name endin with ``*CHANNEL_X_els__d1_*.mmap``.
   c. Two results files, one ending with ``.npz`` and the other with ``.hdf5``.
10. Issues with the run will be written to a file inside ``/pblab/pblab/caiman_scripts``. To view these issues write in the terminal ``cat /pblab/pblab/caiman_scripts/XXXX.power8.tau.ca.il.ER``, where ``XXXX`` is again the job number. If you have no idea what was the problem please consult someone who might.
11. Assuming everything worked out - back in the VNC session (if you left it), **move** these new files from their current folder back to your data folder. We don't want to cog up the ``pblab/pblab`` disk.
12. You probably want to view the results. Press the "Activies" button on the top left part of your VNC screen, and in the search bar write "caiman". An icon should appear named ``CaImAn GUI``. Click on it, and after a few seconds you should see a dialoge screen that asks you for a results file name. Point it to the file ending with ``.hdf5``. Then it will ask for the data - point it to the ``.mmap`` file. After it loads it up you should see a GUI that let's you inspect the components CaImAn found and even add and remove new ones.
13. Congratulations - you're done.

