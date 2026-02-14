import logging
from typing import List
from backend.config import (CALENDAR_TOKEN_PATH)
from langchain_classic.tools import Tool
from langchain_google_community.calendar.toolkit import CalendarToolkit
from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_google_community.calendar.update_event import CalendarUpdateEvent
from langchain_google_community.calendar.delete_event import CalendarDeleteEvent
from langchain_google_community.calendar.search_events import CalendarSearchEvents
from langchain_google_community.calendar.get_calendars_info import GetCalendarsInfo


## ---------------- Logging ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CALENDAR_TOOLS_CACHE = None