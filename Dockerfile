# Use latest tensorflow runtime as a parent image
FROM tensorflow/tensorflow

# Meta-data
LABEL maintainer="Rasoul Asaee <s.a.asaee@gmail.com>" \
      description="Article tagging model:\
      Libraries, data, and code in one image"

# Set the working directory to /app
WORKDIR /app

# Copy the library and the app into the container at /app
COPY src/api /app/api/
COPY src/lib/tokenizer.pickle /app/lib/
COPY src/lib/model /app/lib/model/

# Install the required libraries
RUN pip install --no-cache-dir --upgrade -r /app/api/requirements_prod.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run uvicorn when container launches
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]