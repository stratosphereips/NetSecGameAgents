import socket
import json

class TcpHandler:
    def __init__(self, host, port):
        """Initialize the TCP Agent."""
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        """Establish a connection with the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print("Connected to server.")
        except Exception as e:
            print(f"Error connecting to server: {e}")

    def send_data(self, data):
        """Send data to the server."""
        try:
            self.socket.sendall(data.encode())
        except Exception as e:
            print(f"Error sending data: {e}")

    def receive_data(self, buffer_size=8192):
        """Receive data from the server."""
        try:
            return self.socket.recv(buffer_size).decode()
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

    def close_connection(self):
        """Close the connection to the server."""
        if self.socket:
            self.socket.close()
            print("Connection closed.")

# Example usage of an agent that registers and is about to start the game
if __name__ == "__main__":
    agent = TcpHandler("localhost", 9000)
    agent.send_data("")
    response = agent.receive_data()
    print(f"Received from server: {response}")
    
    # Register
    message_dict = {'PutNick': "Mari"}
    message_str = json.dumps(message_dict)
    agent.send_data(message_str)
    response = agent.receive_data()
    print(f"Received from server: {response}")

    # Choose side
    message_dict = {'ChooseSide': "Attacker"}
    message_str = json.dumps(message_dict)

    agent.send_data(message_str)
    response = agent.receive_data()
    print(f"Received from server: {response}")

    agent.close_connection()
