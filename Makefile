init:
	virtualenv -p python3.6 venv
	. venv/bin/activate; pip install --upgrade -r requirements.txt;