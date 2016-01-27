import pdb
import traceback

import argparse
import pandas as pd
import sys

import common
from fseconomy import FSEconomy


def do_work(args):
    fse = FSEconomy(args.local, args.ukey, args.skey)

    for col in fse.assignments.columns:
        if 'Unnamed' in col:
            del fse.assignments[col]

    aggregated = fse.get_aggregated_assignments(args.cargo)

    result = pd.DataFrame(columns=['FromIcao', 'ToIcao', 'Amount', 'Pay', 'Assignments', 'PtAssignment', 'All-In',
                                   'MakeModel', 'Location', 'Seats', 'MTOW', 'CruiseSpeed', 'RentalDry', 'RentalWet',
                                   'CraftDistance', 'Distance', 'DryRent', 'WetRent', 'DryEarnings', 'WetEarnings',
                                   'DryRatio', 'WetRatio'])
    index = 0
    for _, row in aggregated.iterrows():
        if not args.min and index >= args.limit:
            break
        best_aircraft = fse.get_best_craft(row['FromIcao'], args.radius)
        if best_aircraft is None:
            continue
        for column in ['MakeModel', 'Location', 'Seats', 'MTOW', 'CruiseSpeed', 'RentalDry', 'RentalWet']:
            row[column] = best_aircraft[column]
        row['aircraft'] = best_aircraft
        best_assignments = fse.get_best_assignments(row)
        if best_assignments is None:
            continue
        row['Amount'] = sum(best_assignments['Amount'])
        row['Pay'] = sum(best_assignments['Pay'])
        row['Assignments'] = str(best_assignments['Amount'].tolist())
        row['CraftDistance'] = fse.get_distance(row['FromIcao'], row['aircraft']['Location'])
        row['Distance'] = fse.get_distance(row['FromIcao'], row['ToIcao'])
        row['DryRent'] = round((row['Distance'] + row['CraftDistance']) * row['aircraft']['RentalDry'] / row['aircraft']['CruiseSpeed'], 2)
        row['WetRent'] = round((row['Distance'] + row['CraftDistance']) * row['aircraft']['RentalWet'] / row['aircraft']['CruiseSpeed'], 2)
        row['DryEarnings'] = common.get_earnings(row, 'DryRent')
        row['WetEarnings'] = common.get_earnings(row, 'WetRent')
        if not row['DryEarnings'] + row['WetEarnings']:
            continue
        row['DryRatio'] = common.get_ratio(row, 'DryEarnings')
        row['WetRatio'] = common.get_ratio(row, 'WetEarnings')
        result.loc[index] = row
        index += 1
        if not args.min:
            continue
        if row['DryEarnings'] > args.min or row['WetEarnings'] > args.min:
            break

    print result.sort_values('DryRatio', ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skey', help='Service key')
    parser.add_argument('--ukey', help='User key')
    parser.add_argument('--limit', help='Limit for search', type=int, default=10)
    parser.add_argument('--radius', help='Radius for aircraft search (nm)', type=int, default=50)
    parser.add_argument('--cargo', help='Search cargo assignments (doesn\'t work correctly now)', action='store_true')
    parser.add_argument('--local', help='Use local dump of assignments instead of update', action='store_true')
    parser.add_argument('--debug', help='Use this key to enable debug breakpoints', action='store_true')
    parser.add_argument('--min', help='Minimum earnings (time consuming)', type=int)
    args = parser.parse_args()
    if not (args.skey or args.ukey):
        raise Exception('You have to provide userkey or service key')

    do_work(args)


if __name__ == "__main__":
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
