import sys
import os
from pyngrok import ngrok, conf

# Check for authtoken in arguments or env
auth_token = os.environ.get("NGROK_AUTH_TOKEN")
if len(sys.argv) > 1:
    auth_token = sys.argv[1]

if auth_token:
    ngrok.set_auth_token(auth_token)
    print("Using provided auth token.")
else:
    print("Warning: No auth token provided. Ngrok might require one for persistent tunnels.")

try:
    # Open a HTTP tunnel on port 5000
    public_url = ngrok.connect(5000).public_url
    print("\n" + "="*50)
    print(f"🚀 Halalificator is live at: {public_url}")
    print("="*50 + "\n")
    
    # Keep the script running
    input("Press Enter to stop the tunnel...")
except Exception as e:
    print(f"\n❌ Error starting ngrok: {e}")
    if "authtoken" in str(e).lower():
        print("\nTip: Get a free auth token at https://dashboard.ngrok.com/get-started/your-authtoken")
        print("Then run: python start_tunnel.py YOUR_TOKEN")
