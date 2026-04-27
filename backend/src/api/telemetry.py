# azure opentelemetry integration
import os
import logging
from azure.monitor.opentelemetry import configure_azure_monitor

# create a dedictaed logger
logger = logging.getLogger("brand-guardian-telemtry")

def setup_telemetry():
    '''
    Initializes Azure Monitor OpenTelemtry
    Tracks http requests, database queries, errors, performance metrics
    sends the data to azure monitor

    autocaptures every api requests
    no need to manually log each endpoint
    '''

    # retrieve the connection string
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    # check if configured
    if not connection_string:
        logger.warning("No instrumentation key found. Telemetry is DISABLED.")
        return
    #configure the azure monitor
    try:
        configure_azure_monitor(
            connection_string=connection_string,
            logger_name="brand-guardian-tracer"
        )
        logger.info("Azure Monitor Tracking Enabled and Connected")
    except Exception as e:
        logger.error(f"Failed to Initialize Azure Monitor :{e}")


'''

without Telemetry:
API is slow -> No idea which part
How many users today? No Visibility

with:
/audit endpoint averages 4.5s (Indexer takes 3.8s)
Error logs show : 12% of audits fail due to YouTube download errors
Metrics show : 450 API calls today, 89% success rate
'''