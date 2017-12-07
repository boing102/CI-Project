#! /usr/bin/env python3

from pytocl.main import main
from data_collection_driver import DataCollectionDriver

if __name__ == '__main__':
    driver = DataCollectionDriver()
    with driver.listener():
        main(driver)
