try:
    from api.app import create_app

    print("Import successful")
    app = create_app()
    print("App created")
except Exception as e:
    import traceback

    traceback.print_exc()
