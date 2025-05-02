# Use an appropriate base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file
COPY best_model.pkl /app/

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["python", "app.py"]