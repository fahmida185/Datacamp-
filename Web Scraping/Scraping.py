# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
html = '''
<html>
  <head>
    <title>Intro HTML</title>
  </head>
  <body>
    <p>Hello World!</p>
    <p>Enjoy DataCamp!</p>
  </body>
</html>
'''
# HTML code string
html = '''
<html>
  <body>
    <div class="class1" id="div1">
      <p class="class2">Visit DataCamp!</p>
    </div>
    <div class="you-are-classy">
      <p class="class2">Keep up the good work!</p>
    </div>
  </body>
</html>
'''
# Print out the class of the second div element
whats_my_class( html )
"""
Adding in attributes isn't supposed to be hard, but helps give character to HTML code and at times how it's rendered online.
"""
<html>
  <head>
    <title>Website Title</title>
    <link rel="stylesheet" type="text/css" href="style.css">
  </head>
  <body>
    <div class="class1" id="div1">
      <p class="class2">
        Visit <a href="http://datacamp.com/">DataCamp</a>!
      </p>
    </div>
    <div class="class1 class3" id="div2">
      <p class="class2">
        Or search for it on <a href="http://www.google.com">Google</a>!
      </p>
    </div>
  </body>
</html>
"""
Both div elements belong to the class class1 (even though the second div element also belongs to class3), and both have an a element as a descendant (in this case, a grandchild). So, this direction points to two href attributes, the first is the desired URL, the second is http://www.google.com.
"""
<html>
  <body>
    <div>
      <p>Good Luck!</p>
      <p>Not here...</p>
    </div>
    <div>
      <p>Where am I?</p>
    </div>
  </body>
</html>
"""
Where am I?
"""
xpath = '/html/body/div[2]/p[1]'
# Fill in the blank
"""
sing double forward-slash notation, assign to the variable xpath a simple XPath string navigating to all paragraph p elements within any HTML code
//table directed to all table elements with the HTML.
"""
xpath = '//p'
"""
xpath an XPath string which will select all span elements whose class attribute equals "span-class"
Remember that //div[@id="uid"] selects all div elements whose id attribute equals uid
"""
# Fill in the blank
xpath = '//span[@class="span-class"]