from pymongo import MongoClient, DESCENDING
import http.server
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from urllib.parse import parse_qs, urlparse

PORT = 8000

class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query = parse_qs(parsed_path.query)

        print('path = {}'.format(self.path))
        print('parsed: path = {}, query = {}'.format(parsed_path.path, query))

        name = query.get("name", [""])[0]
        alias_name = query.get("alias_name", [""])[0]
        tag = query.get("tag", [""])[0]

        artists = self.search_artists(name, alias_name, tag)

        form_html = self.form_HTML(name, alias_name, tag)
        artist_html = self.artist_info_HTML(artists)

        html = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>アーティストの検索</title>
        </head>
        <body>
            {form_html}
            {artist_html}
        </body>
        </html>
        '''
       
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()

        self.wfile.write(html.encode("utf-8"))

    def search_artists(self, name, alias_name, tag):
        clinet = MongoClient()
        db = clinet["test"]
        query = {}

        if name:
            query["name"] = name
        if alias_name:
            query["aliases.name"] = alias_name
        if tag:
            query["tags.value"] = tag

        if not query:
            return []


        return db.artists.find(query).sort("rating.count", DESCENDING).limit(100)


    def form_HTML(self, name, alias_name, tag):
        form = f'''
            <form action="/" method="get">
                <div>
                    <label for="name">名前</label>
                    <input name="name" id="name" value="{name}">
                </div>
                 <div>
                    <label for="alias_name">別名</label>
                    <input name="alias_name" id="alias_name" value="{alias_name}">
                </div>
                 <div>
                    <label for="tag">タグ</label>
                    <input name="tag" id="tag" value="{tag}">
                </div>
                <input type="submit">
            </form>
        '''

        return form

    def artist_info_HTML(self, artists):
        table_body = ""

        for artist in artists:
            alias_names = [ alias.get("name", "") for alias in artist.get("aliases", [])]
            tag_values  = [ tag.get("value", "") for tag in artist.get("tags", [])]
            table_body += f'''
            <tr>
                <td>{artist.get("name", "")}</td>
                <td>{artist.get("area", "")}</td>
                <td>{" / ".join(alias_names)}</td>
                <td>{" / ".join(tag_values)}</td>
            </tr>
            '''
        table = f''' 
            <table border="1">
                <tr>
                    <th>名前</th>
                    <th>活動場所</th>
                    <th>別名</th>
                    <th>タグ</th>
                    {table_body}
                </tr>
            </table>
        '''
        return table

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()

