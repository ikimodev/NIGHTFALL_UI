@app.route('/error')
def trigger_error():
    1/0
    return "Error"
