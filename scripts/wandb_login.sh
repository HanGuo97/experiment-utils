#!/usr/bin/expect -f
 
set PASSWORD [lindex $argv 0]

set timeout -1
 
spawn wandb login
 
expect "Enter your choice"
 
send -- "2\r"
 
expect "Paste an API key from your profile and hit enter"
 
send -- "$PASSWORD\r"
 
expect eof