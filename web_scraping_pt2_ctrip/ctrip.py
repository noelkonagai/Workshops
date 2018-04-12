import requests
import pprint
import json
import argparse
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime, timedelta

class Cookie:

	def __init__(self, url):

		self.url = url
		self.abtestID = ""
		self.combinedID = ""
		self.AspNetSessionID = ""
		self.AspNetSessionSVC = ""

	def get(self):

		r = requests.get(self.url)
		self.abtestID = r.cookies['_abtest_userid']
		self.combinedID = r.cookies['_combined']
		self.AspNetSessionID = r.cookies['ASP.NET_SessionId']
		self.AspNetSessionSVC = r.cookies['ASP.NET_SessionSvc']

class Query:

	def __init__(self, date, departureCity, arrivalCity):

		self.url = "https://www.trip.com/flights/Ajax/First"
		self.date = date
		self.departureCity = departureCity
		self.arrivalCity = arrivalCity
		self.passengerType = 'ADT'
		self.seatClass = 'Y'
		self.quantity = '1'
		self.flightResults = None

	def request(self):

		payload = {
			"FlightWay": "OW",
			"DSeatClass": self.seatClass,
			"DSeatSelect": self.seatClass,
			"ChildType": self.passengerType,
			"Quantity": self.quantity,
			"ChildQty": "0",
			"BabyQty": "0",
			"CurrentSeqNO": "1",
			"DCity": self.departureCity,
			"ACity": self.arrivalCity,
			"DDatePeriod1": self.date,
			"filter_ddate": "@",
			"filter_adate": "@",
			"ptype": self.passengerType,
			"Transfer_Type": "-1",
			"TransNo": "20180410104454364",
			"channel_data_list": '{"TrackingID":null,"ShoppingId":null,"BatchID":0,"AllianceID":null,"CampaignCode":null,"Currency":null,"Amount":null,"Language":null,"SID":null,"DepartureCity":"SHA,,1","ArrivalCity":"TYO,,0","DepartureDate":null,"TurnaroundDate":null,"CabinCode":null,"TripType":null,"Channel":null,"OuID":null}',
			"SearchType": "Sync",
		}

		headers = {
		    'accept': "*/*",
		    'origin': "https://www.trip.com",
		    'x-devtools-emulate-network-conditions-client-id': "9FCEAE94A753AD131460912E7B8BBA99",
		    'x-requested-with': "XMLHttpRequest",
		    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
		    'content-type': "application/x-www-form-urlencoded",
		    'referer': "https://www.trip.com/flights/shanghai-to-tokyo/tickets-sha-tyo/?flighttype=s&dcity=sha&acity=tyo&startdate=" + self.date + "&class=y&quantity=1&searchboxarg=t",
		    'accept-encoding': "gzip, deflate, br",
		    'accept-language': "en-US,en;q=0.9,hu;q=0.8,zh-CN;q=0.7,zh;q=0.6",
		    'cookie': "locale=en_US; ibulocale=en_us; ibulanguage=EN; __utma=1.107020246.1523327238.1523327238.1523327238.1; __utmc=1; __utmz=1.1523327238.1.1.utmcsr=ctrip.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; _RF1=101.231.120.133; _RSG=JYKBwBSbAoFdNPX3J05VL8; _RDG=2893ca20170ee822b61523f4b3da625bc0; _RGUID=fc8a95c2-fa2d-4893-855f-1404c38595e7; _tp_search_latest_channel_name=flights; _bfi=p1%3D10320668088%26p2%3D0%26v1%3D1%26v2%3D0; ASP.NET_SessionId=lvt11voe3ugiege41paahkzx; _abtest_userid=e0c32421-4b76-4e7e-b567-e2f4583d071e; LastSearchDDate=" + self.date + "; _combined=transactionId=20180410102732062; __utmb=1.2.10.1523327238; _bfa=1.1523327238023.47zu50.1.1523327238023.1523327238023.1.2; _bfs=1.2; _uetsid=_uet7658fc8b; history=; ASP.NET_SessionSvc=MTAuMTUuMTI4LjM3fDkwOTB8b3V5YW5nfHwxNTEwNzM5NzM2MDYw; locale=en_US; ibulocale=en_us; ibulanguage=EN; __utma=1.107020246.1523327238.1523327238.1523327238.1; __utmc=1; __utmz=1.1523327238.1.1.utmcsr=ctrip.com|utmccn=(referral)|utmcmd=referral|utmcct=/; _RF1=101.231.120.133; _RSG=JYKBwBSbAoFdNPX3J05VL8; _RDG=2893ca20170ee822b61523f4b3da625bc0; _RGUID=fc8a95c2-fa2d-4893-855f-1404c38595e7; _tp_search_latest_channel_name=flights; ASP.NET_SessionId=lvt11voe3ugiege41paahkzx; _abtest_userid=e0c32421-4b76-4e7e-b567-e2f4583d071e; LastSearchDDate=" + self.date + "; history=; _bfi=p1%3D10320667452%26p2%3D10320667452%26v1%3D3%26v2%3D2; _combined=transactionId=20180410105714415; __utmt=1; __utmb=1.4.10.1523327238; _uetsid=_uet7658fc8b; _bfa=1.1523327238023.47zu50.1.1523327238023.1523327238023.1.4; _bfs=1.4; ASP.NET_SessionSvc=MTAuMTUuMTI4LjM0fDkwOTB8b3V5YW5nfHwxNTA5OTcxOTY2MjUw",
		    'cache-control': "no-cache",
		    }

		response = requests.request("POST", self.url, data=payload, headers=headers)

		if response.status_code != 200:
			print("Error: ", response.status_code)

		json_data = json.loads(response.text)

		flight_results = json_data['FlightIntlAjaxResults']

		# #pprint.pprint(flight_results[0]) ## count by numbers, you can know by length how many results there are

		print("The number of flights found for " + str(self.date) + " between " + str(self.departureCity) + " and " + str(self.arrivalCity) + " is " + str(len(flight_results))) 

		i = 0

		self.flightResults = flight_results

	def get_lowest_price(self):

		data = []
		prices = []
		length = []

		for i in range(len(self.flightResults)):
			prices.append(self.flightResults[i]["flightIntlPolicys"][0]["ViewTotalPrice"])
			length.append(self.flightResults[i]["allTime"])

		data.append(prices)
		data.append(length)

		index = prices.index(min(prices))

		print("The lowest price for this route on " + str(self.date) + " is " + str(data[0][index]) + " with a flight time of " + str(data[1][index]))

		# print(flight_results[i]["flightIntlPolicys"][0]["PriceInfos"][0]["TotalPrice"]) ##gives the price in USD of the ith listing
		# print(flight_results[i]["flightIntlPolicys"][0]["ViewTotalPrice"]) ##what actually gets displayed on the website (the cheapest price)

	def json_results(self):

		df = json_normalize(self.flightResults)
		df.to_csv('all_data.csv')

		with open('all_data.json', 'w') as outfile:
			json.dump(self.flightResults, outfile)


def main():

	################################################################
	#
	# COOKIE SESSION
	#
	# Run these to get a session with cookie information. To make
	# sure this works, before you start tweaking the code, add the
	# search URL from your browser below.
	#
	################################################################

	session = Cookie("https://www.trip.com/flights/shanghai-to-tokyo/tickets-sha-tyo/?flighttype=s&dcity=sha&acity=tyo&startdate=2018-08-01&class=y&quantity=1&searchboxarg=t")
	session.get()

	################################################################
	#
	# DEFAULT VALUES
	#
	# This picks a random starting day that's at least a day 
	# away from today, and maximum 100 days away from today.
	#
	################################################################

	today = datetime.strptime(datetime.now().strftime('%Y-%m-%d'), "%Y-%m-%d")
	modified_date = today + timedelta(days=np.random.randint(1,100))
	default_date = datetime.strftime(modified_date, "%Y-%m-%d")

	default_departure_city = 'SHA'
	default_arrival_city = 'TYO'

	################################################################
	#
	# ARGUMENTS
	#
	# As you develop further this program, you might add different
	# arguments. For now, this program can only check prices of
	# one-way tickets.
	#
	################################################################

	parser = argparse.ArgumentParser()

	parser.add_argument('-d', '--date', dest='date', default = default_date, type=str)
	parser.add_argument('-dep', '--departureCity', dest='departure_city', default =default_departure_city, type=str)
	parser.add_argument('-arr', '--arrivalCity', dest='arrival_city', default =default_arrival_city, type=str)

	input_values = parser.parse_args()

	flights = Query(input_values.date, input_values.departure_city, input_values.arrival_city)
	flights.request()
	flights.get_lowest_price()
	flights.json_results()

if __name__ == '__main__':
    main()