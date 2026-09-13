import re
import html
import urllib.parse
import json
from pathlib import Path



class IRealProcessor:
    def __init__(self, input_file):
        self.input_file = Path(input_file)
    # ======================
    # Extract ireal links
    # ======================
    def extract_links(self):
        text = self.input_file.read_text(encoding="utf-8",errors="ignore")
        start = text.find("irealb://")
        if start == -1:
           raise ValueError("No iReal data found")
        data = text[start:]
        songs = data.split("===")
        links = []
        for song in songs:
           song = song.strip()
           if not song.startswith("irealb://"):
              song = "irealb://" + song
           links.append(song)
        return links

    # ======================
    # Decode song
    # ======================

    def decode_song(self,link):
        # html decode
        raw_url = link
        link = html.unescape(link)
        # URL decode
        data = urllib.parse.unquote(link)
        if data.startswith("irealb://"):
           data = data.replace("irealb://","",1)
        parts=data.split("=")
        song={
           "ireal_url": raw_url,
            "song_info":{
                "title":parts[0].strip(),
                "composer":parts[1].strip()
            },
            "style":{
                "feel":parts[3].strip()
            },
            "harmony":{
                "key":parts[4].strip(),
                "raw_chord_data":(parts[6]).strip()
            }
        }
        return song

    # ======================
    # Process all songs
    # ======================

    def process(self):
      data=self.extract_links()
    # # URL decode
    #   data=urllib.parse.unquote(data)

    # # 每首歌之间通过 === 分割
    #   raw_songs=data.split("===")
    #   print("Raw songs:",len(raw_songs))
    #   songs=[]
      links = self.extract_links()
      print("Raw songs:", len(links))
      songs=[]
      for link in links:
          try:
            song=self.decode_song(link)
            songs.append(song)
          except Exception as e:
            print("Skip:",link[:80],e)

      print("Processed songs:",len(songs))
      return songs

    # ======================
    # Save
    # ======================

    def save(self,output):
        songs=self.process()
        output=Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        with open(output,"w",encoding="utf-8") as f:
            json.dump(songs,f,indent=4,ensure_ascii=False)

        print("Saved:",output)


if __name__=="__main__":

    processor=IRealProcessor( "/Volumes/My Passport/Jazz Gen/data/ireal_pro_data/Jazz 1400.ireal")

    processor.save("/Volumes/My Passport/Jazz Gen/data/processed/ireal_json/jazz1400.json")
