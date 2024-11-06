import requests
import argparse


def upload_file(file_path, url):
    with open(file_path, "rb") as file:
        files = {"file": (file_path, file)}
        response = requests.post(url, files=files)

    print(response.json())


def main():
    parser = argparse.ArgumentParser(description="Upload a file to a FastAPI server.")
    parser.add_argument("--train", help="Path to .csv file with training data")
    parser.add_argument("--test", help="Path to .csv file with test data")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:80/",
                        help="URL of the FastAPI file upload endpoint (default: http://127.0.0.1:80/)")

    args = parser.parse_args()
    print(args.train)
    print(args.test)
    print(args.server)

    upload_file(args.train, args.server + 'uploadtrain')
    upload_file(args.test, args.server + 'uploadtest')


if __name__ == "__main__":
    main()
