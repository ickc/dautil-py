REDIRECT = """<!DOCTYPE HTML>
<html lang="en-US">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="0; url={url}">
        <script type="text/javascript">
            window.location.href = "{url}"
        </script>
        <title>Page Redirection</title>
    </head>
    <body>
        <!-- Note: don't tell people to `click` the link, just tell them that it is a link. -->
        If you are not redirected automatically, follow this <a href='{url}'>link</a>.
    </body>
</html>"""


def redirect_html(url):
    '''generate HTML that redirects to url

    c.f. https://stackoverflow.com/questions/5411538/redirect-from-an-html-page
    '''
    return REDIRECT.format(url=url)
