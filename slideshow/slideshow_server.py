import asyncio
from pathlib import Path

from sanic import Sanic
from sanic.response import html, text

slideshow_root_path = Path(__file__).parent

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)

USERS = set()  # saves the currently opened and active websockets
CLOSED = set()  # to temporarily save the websockets which have State.CLOSED


@app.route("/")
async def index(request):
    return html(open(slideshow_root_path / "slideshow.html", "r").read())


@app.route("/send_event")
async def send_event(request):
    """
    A route for emitting an event to all active websockets
    :param request: Contains an event, e.g. "rotate" or "right"
    :return: Success message with the received event
    """
    global USERS
    global CLOSED
    event = request.args["event"][0]  # get event from the get-request
    for u in USERS:  # iterate through the currently opened websockets
        if u.connection.state == 3:  # State.CLOSED
            CLOSED.add(u)  # add closed websocket to CLOSED
            continue
        await u.send(event)  # only send the event to websockets which are still active
    USERS = USERS - CLOSED  # remove closed websockets from USERS
    CLOSED.clear()  # clean-up CLOSED
    return text(f"Received event: {event}")


@app.websocket("/events")
async def emit(_request, ws):
    global USERS
    USERS.add(ws)  # add newly opened websocket connections to USERS
    print("websocket connection opened")
    while True:
        await asyncio.sleep(2)


def main():
    app.run(host="localhost", debug=False)


if __name__ == "__main__":
    main()
