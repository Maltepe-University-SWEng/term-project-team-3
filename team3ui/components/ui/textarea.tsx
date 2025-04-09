import * as React from "react"

import {cn} from "@/lib/utils"

function Textarea({className, ...props}: React.ComponentProps<"textarea">) {
    return (
        <textarea
            data-slot="textarea"
            className={cn(
                "" +
                " placeholder:text-muted-foreground  aria-invalid:ring-destructive/20  aria-invalid:border-destructive flex field-sizing-content max-h-32 overflow-y-auto w-full rounded-md  bg-transparent px-3 py-2 text-base  transition-[color,box-shadow] outline-none  disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
                className
            )}
            {...props}
        />
    )
}

export {Textarea}
