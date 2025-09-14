"""
Complete audit of ALL IQFeed API endpoints and methods
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pyiqfeed as iq
from datetime import datetime, time
import inspect
from dotenv import load_dotenv

load_dotenv()

def inspect_connection_class(conn_class, class_name):
    """Inspect all methods in a connection class"""
    print(f"\n{class_name.upper()} CLASS METHODS:")
    print("=" * 60)

    all_methods = [method for method in dir(conn_class) if not method.startswith('_')]
    request_methods = [method for method in all_methods if 'request' in method.lower()]
    other_methods = [method for method in all_methods if 'request' not in method.lower()]

    print(f"Total methods: {len(all_methods)}")
    print(f"Request methods: {len(request_methods)}")
    print(f"Other methods: {len(other_methods)}")

    print(f"\nREQUEST METHODS:")
    print("-" * 30)
    for method in sorted(request_methods):
        try:
            method_obj = getattr(conn_class, method)
            if callable(method_obj):
                sig = inspect.signature(method_obj)
                doc = method_obj.__doc__
                first_doc_line = doc.split('\n')[0].strip() if doc else "No documentation"
                print(f"  {method}{sig}")
                print(f"    -> {first_doc_line}")
        except Exception as e:
            print(f"  {method} -> Error getting signature: {e}")

    print(f"\nOTHER PUBLIC METHODS:")
    print("-" * 30)
    for method in sorted(other_methods):
        try:
            method_obj = getattr(conn_class, method)
            if callable(method_obj):
                sig = inspect.signature(method_obj)
                print(f"  {method}{sig}")
        except:
            print(f"  {method} -> Property/Attribute")

    return request_methods, other_methods

def comprehensive_audit():
    """Complete audit of all pyiqfeed classes and their methods"""
    print("COMPREHENSIVE PYIQFEED API AUDIT")
    print("=" * 80)

    # Define all connection classes to inspect
    connection_classes = [
        ('HistoryConn', iq.HistoryConn),
        ('LookupConn', iq.LookupConn),
        ('QuoteConn', iq.QuoteConn),
        ('BarConn', iq.BarConn),
        ('NewsConn', iq.NewsConn),
        ('AdminConn', iq.AdminConn),
        ('TableConn', iq.TableConn),
        ('FeedConn', iq.FeedConn)
    ]

    all_request_methods = {}

    for class_name, conn_class in connection_classes:
        try:
            request_methods, other_methods = inspect_connection_class(conn_class, class_name)
            all_request_methods[class_name] = request_methods
        except Exception as e:
            print(f"Error inspecting {class_name}: {e}")

    # Summary
    print(f"\nSUMMARY OF ALL REQUEST METHODS:")
    print("=" * 60)
    total_request_methods = 0
    for class_name, methods in all_request_methods.items():
        print(f"{class_name}: {len(methods)} request methods")
        total_request_methods += len(methods)

    print(f"\nTOTAL REQUEST METHODS ACROSS ALL CLASSES: {total_request_methods}")

    return all_request_methods

def audit_all_api_methods():
    """Audit every available IQFeed API method"""
    print("COMPLETE IQFeed API AUDIT")
    print("="*80)

    # Connect to IQFeed
    username = os.getenv('IQFEED_USERNAME', '487854')
    password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    service = iq.FeedService(
        product="FUZZY_OSS20",
        version="1.0",
        login=username,
        password=password
    )
    service.launch(headless=True)

    print("1. HISTORY CONNECTION METHODS:")
    print("-" * 40)
    hist_conn = iq.HistoryConn(name="api-audit")
    hist_conn.connect()

    # Get all methods
    history_methods = [method for method in dir(hist_conn) if not method.startswith('_')]
    print(f"Total HistoryConn methods: {len(history_methods)}")

    for method in sorted(history_methods):
        if 'request' in method.lower():
            print(f"   {method}")
            try:
                method_obj = getattr(hist_conn, method)
                if callable(method_obj):
                    help_text = method_obj.__doc__
                    if help_text:
                        first_line = help_text.split('\n')[0].strip()
                        print(f"     -> {first_line}")
            except:
                pass

    print(f"\n2. TESTING TICK DATA METHODS:")
    print("-" * 40)

    # Test each tick method with AAPL
    tick_methods = [m for m in history_methods if 'tick' in m.lower() and 'request' in m.lower()]

    for method_name in tick_methods:
        print(f"\nTesting {method_name}:")
        try:
            method = getattr(hist_conn, method_name)

            if method_name == 'request_ticks':
                result = method(ticker="AAPL", max_ticks=10)
                print(f"   SUCCESS: Got {len(result) if result else 0} ticks")

            elif method_name == 'request_ticks_for_days':
                # Test 1: No time filter
                result = method(ticker="AAPL", num_days=1, max_ticks=10)
                print(f"   No filter: Got {len(result) if result else 0} ticks")

                # Test 2: Market hours filter
                result = method(ticker="AAPL", num_days=1, bgn_flt=time(9,30), end_flt=time(16,0), max_ticks=10)
                print(f"   Market hours: Got {len(result) if result else 0} ticks")

                # Test 3: Different days
                for days in [1, 2, 5]:
                    result = method(ticker="AAPL", num_days=days, max_ticks=10)
                    print(f"   {days} days: Got {len(result) if result else 0} ticks")

            elif method_name == 'request_ticks_in_period':
                # Test date range
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                result = method(ticker="AAPL", bgn_dt=start_date, end_dt=end_date, max_ticks=10)
                print(f"   Date range: Got {len(result) if result else 0} ticks")

        except Exception as e:
            print(f"   FAILED: {e}")

    print(f"\n3. TESTING BAR DATA METHODS:")
    print("-" * 40)

    bar_methods = [m for m in history_methods if 'bar' in m.lower() and 'request' in m.lower()]

    for method_name in bar_methods:
        print(f"\nTesting {method_name}:")
        try:
            method = getattr(hist_conn, method_name)

            if 'daily' in method_name:
                result = method(ticker="AAPL", num_days=5)
                print(f"   SUCCESS: Got {len(result) if result else 0} daily bars")

            elif 'for_days' in method_name:
                result = method(ticker="AAPL", interval_len=60, interval_type='s', days=1)
                print(f"   1-minute bars: Got {len(result) if result else 0} bars")

                # Test with market hours
                result = method(ticker="AAPL", interval_len=60, interval_type='s', days=1,
                              bgn_flt=time(9,30), end_flt=time(16,0))
                print(f"   Market hours: Got {len(result) if result else 0} bars")

        except Exception as e:
            print(f"   FAILED: {e}")

    print(f"\n4. LOOKUP CONNECTION METHODS:")
    print("-" * 40)
    lookup_conn = iq.LookupConn(name="lookup-audit")
    lookup_conn.connect()

    lookup_methods = [method for method in dir(lookup_conn) if 'request' in method.lower()]
    for method in sorted(lookup_methods):
        print(f"   {method}")

    print(f"\nAPI AUDIT COMPLETE")

if __name__ == "__main__":
    print("Choose audit type:")
    print("1. Comprehensive class inspection (no connection required)")
    print("2. Full API test with connections")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        comprehensive_audit()
    elif choice == "2":
        audit_all_api_methods()
    else:
        print("Running comprehensive audit by default...")
        comprehensive_audit()