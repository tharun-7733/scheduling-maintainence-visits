FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements directly
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose port for Gradio
# EXPOSE 7860

# Run your app (change app.py to your entry file)
CMD ["python", "gr.py"]
