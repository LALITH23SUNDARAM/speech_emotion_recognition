# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.6

EXPOSE 8501

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY app.py .
COPY LSTM_model_weights.h5 .
COPY LSTM_model.json .
COPY train_mean .
COPY train_var .


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT [ "streamlit", "run" ]
CMD ["app.py"]
