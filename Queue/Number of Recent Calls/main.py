class RecentCounter:
    def __init__(self):
        self.pings = []  # List to store the timestamps of pings

    def ping(self, t: int) -> int:
        self.pings.append(t)  # Add the current timestamp to the list
        # Count the number of pings in the last 3000 milliseconds
        while self.pings[0] < t - 3000:
            self.pings.pop(0)  # Remove timestamps that are out of the range
        return len(self.pings)  # Return the count of recent pings