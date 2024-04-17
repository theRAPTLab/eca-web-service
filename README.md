# ECA Web Service

## Running the ECA Locally
* Create a virtual environment within the project: https://flask.palletsprojects.com/en/3.0.x/installation/#virtual-environments
* Activate the virtual environment. You have to do this every time you want to run the project. See the link above for how to activate a virtual environment.
    * The shell prompt will change, confirming that you have activated the environment
    * (Windows) If the virtual environment doesn't activate because of an error that says "running scripts is disabled on this system", enter the following into your terminal:
        `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
    * To learn more about what this does, see this page: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4


* Add a **model** folder in **dialog_agent**. To that, add a folder called **model_ci_textbite**. You can get this from Vikram or Capen. Unzip before adding to the **model** folder.
    * https://drive.google.com/file/d/1rRKH5z3QZAlsmBBNWWcHLDIrlxRf1759/view?usp=drive_link

* Add the `eca_data.csv` that corresponds to the trained model in **dialog_agent/data**. Replace any existing `eca_data.csv`.

* If one does not already exist, add a **log** folder. You should only have to do this once. The .gitignore file exclues file in the folder but keeps the folder.

**Auto-install** package dependencies with `pip install -r requirements.txt`

* (Windows) Type ".\run.sh" in your command line and press enter. If successful, you should see a new bash window open and show that the app is running on `https://127.0.0.1:5000/`
* Navigate to that address in your browser and if you see "Hello, World!" the engine is running.