print("Step 1")
import config.settings
print("Step 2")
import config.logging_config
print("Step 3")
import api.app
print("Step 4")
app = api.app.create_app()
print("Step 5")
