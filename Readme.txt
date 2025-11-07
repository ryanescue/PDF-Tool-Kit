## Features (MVP target)
- [x] Project scaffold (Django, apps, Git)
- [ ] Upload documents (DOCX/XLSX/TXT/JPG/PNG → PDF)
- [ ] Merge & Split PDFs
- [ ] Compare two documents (diff)
- [ ] Digital signatures (basic)
- [ ] User auth: Join / Login / Logout
- [ ] Dashboard: file history & actions
- [ ] `/server_info/` endpoint (per course rubric)
- [ ] Docker + GCP deploy (MIG + LB + SSL, Postgres, nginx+gunicorn)

## Tech
- **Backend:** Python 3, Django
- **Frontend:** Bootstrap 5
- **Storage:** SQLite (dev), Postgres (prod)
- **Deploy:** Docker, GCP (Managed Instance Group + Load Balancer), nginx + gunicorn

## Project Structure

DOCUMENT_EXTRACTOR/ Root
- core/ # Django project config (settings/urls/wsgi/asgi)
- accounts/ # join/login/logout/profile, templates/registration/*
- pdf_toolkit/ # uploads & all PDF operations, dashboard
│  - models.py # Document, OperationJob, Signature, Comparison, Folder
│  - views/ # split per feature (convert.py, merge.py, etc.) [optional]
│  - urls.py
│  - templates/pdf_toolkit/ # home, upload, merge, split, compare, sign, dashboard
│  - static/pdf_toolkit/
- templates/ # project-level base.html, navbar, about.html (optional)
- static/ # project-level CSS/JS (optional)
- manage.py
- requirements.txt

##thoughts?

Was thinking this let me know what you think ^ , Created it also used Chat to help us with a to-do list we can add or delete parts of it but i think it would be nice to follow to maybe split stuff. 

Have nothing in the static, could be the css, js, images if we want

I have not set up docker

11/6/2025


## Local Setup

```bash
Windows (PowerShell)
    python -m venv venv
    venv\Scripts\activate

pip install django requests gunicorn

