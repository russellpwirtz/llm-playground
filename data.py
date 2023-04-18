# Import the csv module
import csv
# Import the datetime module
import datetime


def parse_csv(symbol: str, name: str = 'data.csv') -> list:
    # Open the csv file in read mode
    with open("data.csv", "r") as f:
        # Create a csv reader object
        reader = csv.reader(f)
        # Loop through each row in the file
        result = []
        for row in reader:
            # Check if the row has six elements
            if len(row) == 6:
                inner_result = {}
                inner_result['symbol'] = symbol
                # Convert the first four elements to floats
                inner_result['ohlc'] = [float(x) for x in row[:4]]
                # Convert the fifth element to an integer
                inner_result['timestamp'] = int(row[4])
                # Convert the sixth element to a datetime object
                inner_result['date'] = row[5]
                # Print the parsed data
                print(str(inner_result))
                result.append(inner_result)
            else:
                # Skip the row or handle it differently
                print("Invalid row:", row)
                # raise Exception("Invalid csv row: " + str(row))
        return result


parse_csv("ABCD")
