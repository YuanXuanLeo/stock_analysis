$(function(){
    $("button").on("click", loadServerData);
});

function loadServerData(){
    //data -> items(array)
    //title, link, pubDate(空格切割,取前面)
    let rss2json = "https://api.rss2json.com/v1/api.json?rss_url=";
    $.getJSON(rss2json+"https://tw.stock.yahoo.com/rss?q=%E8%99%9B%E6%93%AC%E8%B2%A8%E5%B9%A3")
    .done(function(data){
        // debugger;
        for(let x=0;x<data.items.length;x++){
            $("#dataTable").append(
                `<tr><td><a target='_blank' href='${data.items[x].link}'>${data.items[x].title}</a></td><td>${data.items[x].pubDate.split(" ")[0]}</td></tr>`
            );
        }
    })
    .fail(function(){console.log("Error");})
    .always(function(){console.log("Always");})
}