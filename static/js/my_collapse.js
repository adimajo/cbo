
function set_collapsible(){
    var coll = document.getElementsByClassName("collapsible");
    var i;
    for (i = 0; i < coll.length; i++)
        {
        coll[i].addEventListener("click", function()
        {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.maxHeight)    {
                content.style.maxHeight = null;
        } else {
            if (content.classList.contains("tall")) {
                content.style.maxHeight = "3500px";
                content.style.minHeight = "2000px";
            } else {
                content.style.maxHeight = content.scrollHeight + 20 + "px";
            }
        }
        });
        }
}
set_collapsible();








