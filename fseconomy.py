import pandas as pd
import pickle
import time
import numpy as np
import requests
import requests_cache
from math import radians

from pulp import LpMaximize, LpProblem, LpVariable
from StringIO import StringIO
from requests_throttler import BaseThrottler

import common
import const


class FSEconomy(object):
    def __init__(self, local, user_key=None, service_key=None):
        requests_cache.install_cache('fse', expire_after=3600)
        self.bt = BaseThrottler(name='fse-throttler', reqs_over_time=(1, 2))
        self.bt.start()
        self.airports = common.load_airports()
        self.aircraft = common.load_aircrafts()
        self.service_key = service_key
        self.user_key = user_key
        if local:
            self.assignments = common.load_pickled_assignments()
        else:
            self.assignments = self.get_assignments()

    def get_aggregated_assignments(self, cargo=False):
        self.assignments = self.assignments[self.assignments.UnitType == 'kg']
        # if cargo:
        #     self.assignments = self.assignments[self.assignments.UnitType == 'kg']
        # else:
        #     self.assignments = self.assignments[self.assignments.UnitType == 'passengers']
        grouped = self.assignments.groupby(['FromIcao', 'ToIcao', 'UnitType'], as_index=False)
        aggregated = grouped.aggregate(np.sum)
        aggregated = aggregated.sort_values('Pay', ascending=False)
        return aggregated

    def send_single_request(self, path):
        query_link = self.generate_request(const.LINK + path)
        request = requests.Request(method='GET', url=query_link)
        i = 0
        while True:
            try:
                thottled_request = self.bt.submit(request)
                data = thottled_request.response.content
                if 'To many requests' in data or 'minimum delay' in data:
                    raise requests.exceptions.ConnectionError
                return data
            except requests.exceptions.ConnectionError:
                requests_cache.clear()
                if i >= 10:
                    raise
                print 'Retrying Request'
                i += 1
                time.sleep(2)

    def send_multi_request(self, paths):
        request_queue = []
        for path in paths:
            query_link = self.generate_request(const.LINK + path)
            request_queue.append(requests.Request(method='GET', url=query_link))

        i = 0
        while True:
            try:
                thottled_requests = self.bt.multi_submit(request_queue)
                responses = [tr.response for tr in thottled_requests]
                request_queue = []
                complete_response = []
                for response in responses:
                    if 'To many requests' in response.content or 'minimum delay' in response.content:
                        request_queue.append(response.url)
                        print response.content
                    elif 'you are now in a lockout period' in response.content:
                        raise Exception(response.content)
                    else:
                        complete_response.append(response.content)

                if len(request_queue) > 0:
                    raise requests.exceptions.ConnectionError

                return complete_response
            except AttributeError:
                for request in request_queue:
                    print 'Error with request: ', request
                raise
            except requests.exceptions.ConnectionError:
                requests_cache.clear()
                if i >= 10:
                    raise
                print 'Retrying Request'
                i += 1
                time.sleep(2)

    def get_aircrafts_by_icaos(self, icaos):
        aircraft_requests = []
        for icao in icaos:
            aircraft_requests.append('query=icao&search=aircraft&icao={}'.format(icao))

        responses = self.send_multi_request(aircraft_requests)
        all_aircraft = []
        for response in responses:
            aircraft = pd.DataFrame.from_csv(StringIO(response))
            try:
                aircraft.RentalDry = aircraft.RentalDry.astype(float)
                aircraft.RentalWet = aircraft.RentalWet.astype(float)
                all_aircraft.append(aircraft)
            except:
                print 'error updating rental info: ', response

        return all_aircraft

    def get_assignments(self):
        assignments = pd.DataFrame()

        i = 0
        assignment_requests = []
        number_at_a_time = 1000
        while i + number_at_a_time < len(self.airports):
            assignment_requests.append(
                'query=icao&search=jobsfrom&icaos={}'.format('-'.join(self.airports.icao[i:i + number_at_a_time])))
            i += number_at_a_time

        f = open('assignments.csv', 'a')
        responses = self.send_multi_request(assignment_requests)
        for data in responses:
            f.write(data)
            assignments = pd.concat([assignments, pd.DataFrame.from_csv(StringIO(data))])

        response = self.send_single_request('query=icao&search=jobsfrom&icaos={}'.format('-'.join(self.airports.icao[i:len(self.airports) - 1])))
        assignments = pd.concat([assignments, pd.DataFrame.from_csv(StringIO(response))])
        f.write(response)
        with open('assignments', 'wb') as f:
            pickle.dump(assignments, f)
        return assignments

    def max_fuel(self, aircraft):
        return aircraft['Ext1'] + aircraft['LTip'] + aircraft['LAux'] + aircraft['LMain'] + aircraft['Center1'] + aircraft['Center2'] + aircraft['Center3'] + aircraft['RMain'] + aircraft['RAux'] + aircraft['RTip'] + aircraft['RExt2']

    def estimated_fuel(self, distance, aircraft):
        # Add 1.5 hours
        return (((round(distance / aircraft['CruiseSpeed'], 1)) * aircraft['GPH']) + (aircraft['GPH'] * 1.5)) * 0.81

    def get_best_assignments(self, row):
        max_cargo = 0
        if row['UnitType'] == 'passengers':
            df = self.assignments[(self.assignments.FromIcao == row['FromIcao']) &
                                  (self.assignments.ToIcao == row['ToIcao']) &
                                  (self.assignments.Amount <= row['Seats']) &
                                  (self.assignments.UnitType == 'passengers')]
        else:
            distance = self.get_distance(row['FromIcao'], row['ToIcao'])
            max_cargo = round(row['aircraft']['MTOW'] - row['aircraft']['EmptyWeight'] - self.estimated_fuel(distance, row['aircraft']))
            if max_cargo <= 0:
                return None
            df = self.assignments[(self.assignments.FromIcao == row['FromIcao']) &
                                  (self.assignments.ToIcao == row['ToIcao']) &
                                  (self.assignments.Amount <= max_cargo) &
                                  (self.assignments.UnitType == 'kg')]
        if not len(df):
            return None
        prob = LpProblem("Knapsack problem", LpMaximize)
        w_list = df.Amount.tolist()
        p_list = df.Pay.tolist()
        x_list = [LpVariable('x{}'.format(i), 0, 1, 'Integer') for i in range(1, 1 + len(w_list))]
        prob += sum([x * p for x, p in zip(x_list, p_list)]), 'obj'
        if (row['UnitType'] == 'passengers'):
            prob += sum([x * w for x, w in zip(x_list, w_list)]) <= row['Seats'], 'c1'
        else:
            prob += sum([x * w for x, w in zip(x_list, w_list)]) <= row['aircraft']['MTOW'], 'c1'
        prob.solve()
        return df.iloc[[i for i in range(len(x_list)) if x_list[i].varValue]]

    def get_best_craft(self, icao, radius):
        print 'Searching for the best aircraft from {}'.format(icao)
        max_mtow = 0
        best_aircraft = None
        near_icaos = self.get_closest_airports(icao, radius).icao

        all_aircraft = self.get_aircrafts_by_icaos(near_icaos)
        for aircraft in all_aircraft:
            if not len(aircraft):
                continue
            merged = pd.DataFrame.merge(aircraft, self.aircraft, left_on='MakeModel', right_on='Model', how='inner')
            merged = merged[
                (~merged.MakeModel.isin(const.IGNORED_AIRCRAFTS)) & (merged.RentalWet + merged.RentalDry > 0)]
            if not len(merged):
                continue
            aircraft = merged.ix[merged.MTOW.idxmax()]
            if aircraft.MTOW > max_mtow:
                best_aircraft = aircraft
                max_mtow = aircraft.MTOW
        return best_aircraft

    def get_closest_airports(self, icao, nm):
        lat = self.airports[self.airports.icao == icao].lat.iloc[0]
        nm = float(nm)
        # one degree of latitude is appr. 69 nm
        lat_min = lat - nm / 69
        lat_max = lat + nm / 69
        filtered_airports = self.airports[self.airports.lat > lat_min]
        filtered_airports = filtered_airports[filtered_airports.lat < lat_max]
        distance_vector = filtered_airports.icao.map(lambda x: self.get_distance(icao, x))
        return filtered_airports[distance_vector < nm]

    def get_distance(self, from_icao, to_icao):
        try:
            lat1, lon1 = [radians(x) for x in self.airports[self.airports.icao == from_icao][['lat', 'lon']].iloc[0]]
            lat2, lon2 = [radians(x) for x in self.airports[self.airports.icao == to_icao][['lat', 'lon']].iloc[0]]
        except IndexError:
            return 9999.9
        return common.get_distance(lat1, lon1, lat2, lon2)

    def get_logs(self, from_id):
        key = self.user_key or self.service_key
        data = self.send_single_request('query=flightlogs&search=id&readaccesskey={}&fromid={}'.format(key, from_id))
        logs = pd.DataFrame.from_csv(StringIO(data))
        logs = logs[(logs.MakeModel != 'Airbus A321') & (logs.MakeModel != 'Boeing 737-800') & (logs.Type == 'flight')]
        logs['Distance'] = logs.apply(lambda x, self=self: self.get_distance(x['From'], x['To']), axis=1)
        logs = pd.merge(logs, self.aircraft, left_on='MakeModel', right_on='Model')
        logs['FlightTimeH'] = logs.apply(lambda x: int(x['FlightTime'].split(':')[0]), axis=1)
        logs['FlightTimeM'] = logs.apply(lambda x: int(x['FlightTime'].split(':')[1]), axis=1)
        logs = logs[(logs.FlightTimeH > 0) | (logs.FlightTimeM > 0)]
        logs = logs[logs.Distance > 0]
        logs['AvSpeed'] = logs.apply(lambda x: 60 * x['Distance'] / (60 * x['FlightTimeH'] + x['FlightTimeM']), axis=1)
        import pdb
        pdb.set_trace()

    def generate_request(self, query_link):
        if self.user_key:
            query_link += '&userkey={}'.format(self.user_key)
        elif self.service_key:
            query_link += '&servicekey={}'.format(self.service_key)
        return query_link

    def __del__(self):
        self.bt.shutdown()
