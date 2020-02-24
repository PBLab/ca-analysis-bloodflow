-----------------------------
Running the Analysis Pipeline
-----------------------------

As you may already know, this repository contains functions desgined to analyze calcium and vascular data taken by our lab.
The way to run these scripts is through an automated web-based GUI called "Helium" that runs the underlying
functions and algorithms in their correct order, recording the results of each step in a database.

Principles of operation
-----------------------
When you have new calcium data you wish to analyze, your first goal is to record it in the database. This process involves
pointing the database to the relevant folder and entering several required paramateres of the acquisition so that
the analysis would be correct and repeatable. This process is done manually on an experiment-by-experiment basis, and
can be performed by all lab members.

After the initial parameters were entered the user needs not to be involved any further. The server will recognize that
a new dataset was entered and will start running the analysis pipeline independently. Progress can be monitored by looking
at the data tables and viewing the updated rows. Errors will also be reported there. The analyzed data will be located
in the hard-drive, in the same folders in which the original dataset is located.

Usage
-----
To run the pipeline, you have to connect to our lab's server (``cortex.tau.ac.il``) and open the GUI from there.

1. ``ssh -X username@cortex.tau.ac.il`` - ``username`` is your Cortex username.
2. Type in your password.
3. Open up firefox by typing in the terminal ``firefox``.
4. In the address bar, type the following address:
5. You should see the purple Helium home screen, asking for your credentials.
6. Username: ``root``. Host: ``172.17.0.2``. Password: our standard lab password.
7. You should see an "empty" screen with a menu icon in the top left. Click it. It will open the different data tables that can be found in our database. Choose the "" table.
8. Edit it to enter a new row. Please be careful - don't edit any other rows.

:: warning:
    We were unable to control the permissions so that a user would only be able to enter a new row. Unfortunately each of you can completely erase
    everything in the database if so you wish. So please be extra careful when you edit the table.


