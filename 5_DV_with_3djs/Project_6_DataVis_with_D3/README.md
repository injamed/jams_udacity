##Summary

[First version of visualization](http://bl.ocks.org/injamed/raw/075680bdbebc6bcb91f97ed38890a72e/)

[Second version](http://bl.ocks.org/injamed/raw/5bd91da5c55993f4acd8df169a46b1cc/)

[Current final version](http://bl.ocks.org/injamed/f743d63f68c7215d52749a250f59c2e3)

One year ago I've moved from Ukraine to USA. Using the data from [databank](http://databank.worldbank.org),
I've built a tool to "compare" countries by most interesting (in my opinion) 
and most populated by numbers indicators. Every "section" of the interactive
visualization can lead to its own conclusion which appears at the bottom 
of the chart. Overall verdict is declared at the top of the page.

##Design
I've used line charts, which are the best for the time series data; colors to
encode "goodness" property of the chosen indicator, I believe, this can help
readers to be aware, what to look for (are higher values better or worse); also,
I've added the newest available data in inscriptions, so it is straightforward
to compare the current situation.
After receiving the feedback, I've:

1. (first iteration) changed design of the original drop-down list
to be "extendable on demand" to not more, than 5 items, because unfolding
drop-down prevented readers to see the content on the page

2. used color hues to mark country with higher value of indicator with more 
saturated color

3. fixed a description of the "newest available data" a little

4. (second iteration) changed the structure in narration to "one chart - 
one conclusion", which can help users to compare the conclusion with the 
graphical representation

5. pointed out the main conclusion on the top of the page

6. moved the drop-down above the graph, to serve as title to the graph.

I hope that the current version is fitting to the screen, so there is no need
to scroll down to find an appropriate piece of conclusion. 
Below there is a suggestion to fit all visualization on one page. Well, it is
really a good idea, but, in my opinion, many visualization on one page will be
hard to read. Also, in this case I would have to abandon an idea to show numbers
on the contours of the countries, and I cherish this idea as the author)

##Feedback

(1st)

>Overall, I think you have some good visualizations of some interesting data. 
>A couple of things that could use some refinement:
>
>1. I didn't have my browser window maximized when I clicked the link, and the 
>data visualization didn't fit in the window until I maximized it. 
>So, resizing of the visualization based on window size would be nice.
>
>2. It took me a while to realize that the numbers printed on the country 
>silhouettes were the respective values for 2013. Doing something to make this 
>a little more obvious would be nice. You might also want to mention that the 
>data only goes up to 2013 and that's why that's the value that is shown.
>
>3. The drop down list of options seems a little overwhelming. There is a lot 
>of data in this visualization, but not much guidance or narrative for the user.
> So, even though there is a lot of information you can get out of the 
>visualization, I'm left wondering if there's a point you're trying to make or 
>if I'm just looking at a bunch of charts.

(2nd)

>Excellent work!!
>
>
>What I noticed (combined with relationships that I noticed):
>
>The most obvious feature of this visualization is that it provides 
>a clear way to compare two countries.
>
>The data for each country is provided in two ways, a simple visual 
>for a single year and a graph for historical data
>
>The relationships described in the text are clearly shown in the visualization.
>
>
>What questions I have about the data:
>
>None, the description of the data sources, as well as a link 
>(if I want more detail), is provided.
>
>
>Is there something you don’t understand in the graphic?:
>
>No, as I say, it is clearly described and the graphics correspond to the 
>description.
>
>
>The only thing that distracts from the visualization is that is needs to be 
>styled better. Using CSS to ensure that the visualization occupies the screen 
>would make this a perfect visualization (currently, I have to scroll down to 
>access the dropdown menu and scroll back up to see the result).
>
>Overall, a really nice job!!

(3rd)

>Great idea for your visualization! Here is some feedback:
>
>What do you notice in the visualization?
>My first reaction to the visualization was Wow, I love the concept of comparing 
>the two countries like this and showing them visually really adds to the 
>experience. There is a bit of disconnect between the countries and the chart, 
>which can be confusing (where should I be looking for the information?), but 
>there is alot of beauty in what you put together.
>
>As Myles said, the scale could be adjusted to make the important information 
>easier to see quickly, without having to scroll (I am struggling with this in 
>my own visualization!).
>
>Also, I was a bit overwhelmed by all of the options in the drop menu. I liked 
>having the option to poke around, but didn't have a reason to until I saw your 
>conclusions on the bottom. If you were able merge some of your conclusions into 
>a story which you then highlight with only the relevant options in the drop 
>down, I think your visualization would be great!
>
>What questions do you have about the data?
>My main questions are about the data story and purpose of the comparison. I 
>like the conclusions you came to, but may not have found them on my own. Since 
>this is supposed to be explanatory, I think you can steer your readers as to 
>what they are looking at. Once the story was in my head and I start looking 
>through the data, I began to understand better what the purpose of the 
>visualization was.
>
>What relationships do you notice?
>The comparisons between the countries in these categories is great. I was able 
>to see clearly the difference between the countries for each category.
>
>What do you think is the main takeaway from this visualization?
>I don't see a main takeaway, as there are so many comparisons between the 
>countries. I think refining a data story would help with this.
>
>Is there something you don’t understand in the graphic?
>I was initially confused with the colors assigned to the countries, but then 
>understood that if they are colored blue, then the higher number was better, 
>etc. Would you be able to highlight the "better" country, instead of coloring 
>them both at the same time? For example, when CO2 Emissions is selected, could 
>you only highlight Ukraine to show that it is much better than the US in this 
>category? I think it would help the reader more quickly identify the difference 
>between the countries.
>
>My concluding thoughts are that I think you could swap the data source and note 
>with your conclusions or story, since those are more important to understanding 
>the visualization. Also, I'm not sure if it would be beneficial to add to the 
>visualization, but I found myself wondering what the difference was in GDP 
>between the countries. This might help frame some of the conclusions.

(4th)

>I love your visualisation. Here's my feedback.
>
>What do you notice in the visualisation? 
>Ukraine and US stats vary considerably. 
>Very clear comparison of economic drivers of both countries in all areas.
>Do you have any questions about the data? No, very comprehensive.
>What relationships do you notice? 
>Various, I like the way you have listed your own observations 
>at the end which can be a good starting point for digging deeper into the data.
>What do you think is the main takeaway from this visualisation? All said above.
>Is there something you don’t understand? 
>No, very clear and to the point.>
>
>The only think I'd say is it would be nice if you could fit all the visuals 
>on one page? Tricky I know having struggled with it myself. 
>Many thanks for letting me see your project.

##Resources

As a part of my visualization, I've used the code for the legend, [source](http://bl.ocks.org/ZJONSSON/3918369)
And, in general, a stackoverflow. In many questions concerning JavaScript. 